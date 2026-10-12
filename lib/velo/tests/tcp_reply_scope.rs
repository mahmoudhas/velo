// SPDX-FileCopyrightText: Copyright (c) 2026 Velo contributors.
// SPDX-License-Identifier: Apache-2.0

//! Synthetic, loopback-only probes with deliberately duplicate wire identities.
//! A request's reply must stay on its own accepted connection, not whichever
//! connection most recently claimed the same worker or even the same response ID.
use bytes::Bytes;
use futures::StreamExt;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpStream, sync::Semaphore};
use tokio_util::codec::Framed;
use velo::transports::{
    MessageType, ShutdownPolicy,
    tcp::{TcpFrameCodec, TcpTransportBuilder},
};
use velo::{Handler, Messenger};

type Socket = Framed<TcpStream, TcpFrameCodec>;
const ID: u128 = (1 << 80) | 42;

fn header(id: u128, kind: u8, handler: &[u8], metadata: &[u8]) -> Vec<u8> {
    let mut bytes = vec![1, kind];
    bytes.extend_from_slice(&id.to_le_bytes());
    bytes.extend_from_slice(&(handler.len() as u16).to_le_bytes());
    bytes.extend_from_slice(handler);
    bytes.extend_from_slice(&(metadata.len() as u16).to_le_bytes());
    bytes.extend_from_slice(metadata);
    bytes
}

async fn send(socket: &mut Socket, header: &[u8], payload: &[u8]) {
    TcpFrameCodec::encode_frame(socket.get_mut(), MessageType::Message, header, payload)
        .await
        .unwrap();
}

async fn receive(socket: &mut Socket) -> (MessageType, Bytes, Bytes) {
    tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("reply did not reach its request connection")
        .expect("connection closed")
        .unwrap()
}

struct Fixture {
    node: Arc<Messenger>,
    address: SocketAddr,
    started: flume::Receiver<Bytes>,
    barrier: flume::Receiver<()>,
    release: Arc<Semaphore>,
}

impl Fixture {
    async fn new() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let tcp = TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap();
        let node = Messenger::builder()
            .add_transport(Arc::new(tcp))
            .build()
            .await
            .unwrap();
        let (started_tx, started) = flume::unbounded();
        let (barrier_tx, barrier) = flume::unbounded();
        let release = Arc::new(Semaphore::new(0));
        let gate = release.clone();
        node.register_handler(
            Handler::unary_handler_async("held", move |ctx| {
                let gate = gate.clone();
                let started = started_tx.clone();
                async move {
                    started.send(ctx.payload.clone()).unwrap();
                    gate.acquire().await.unwrap().forget();
                    Ok(Some(ctx.payload))
                }
            })
            .build(),
        )
        .unwrap();
        node.register_handler(
            Handler::am_handler("barrier", move |_| {
                barrier_tx.send(()).unwrap();
                Ok(())
            })
            .build(),
        )
        .unwrap();
        Self {
            node,
            address,
            started,
            barrier,
            release,
        }
    }

    async fn connect(&self) -> Socket {
        Framed::new(
            TcpStream::connect(self.address).await.unwrap(),
            TcpFrameCodec::new(),
        )
    }

    async fn pending(&self, socket: &mut Socket, id: u128, payload: &'static [u8]) {
        send(socket, &header(id, 2, b"held", b""), payload).await;
        let started = tokio::time::timeout(Duration::from_secs(2), self.started.recv_async())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(started.as_ref(), payload);
    }

    async fn finish(self) {
        self.node
            .graceful_shutdown(ShutdownPolicy::WaitForever)
            .await;
    }
}

/// Barrier is a different worker: it proves all preceding malformed frames on
/// this socket reached the dispatcher, without overwriting the victim's route.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejected_headers_cannot_redirect_a_pending_reply() {
    let valid = header(ID, 2, b"held", b"");
    let mut bad_name_length = valid.clone();
    bad_name_length[18..20].copy_from_slice(&u16::MAX.to_le_bytes());
    let mut bad_metadata_length = valid.clone();
    let n = bad_metadata_length.len();
    bad_metadata_length[n - 2..].copy_from_slice(&8u16.to_le_bytes());
    let cases = [
        valid[..18].to_vec(),
        bad_name_length,
        header(ID, 2, b"", b""),
        header(ID, 2, &[0xff], b""),
        bad_metadata_length,
        header(ID, 2, b"held", &[0xc1]),
        header(ID, 99, b"held", b""),
    ];
    for malformed in cases {
        let f = Fixture::new().await;
        let mut a = f.connect().await;
        let mut b = f.connect().await;
        f.pending(&mut a, ID, b"private-a").await;
        send(&mut b, &malformed, b"").await;
        send(&mut b, &header(ID + 1, 0, b"barrier", b""), b"").await;
        tokio::time::timeout(Duration::from_secs(2), f.barrier.recv_async())
            .await
            .unwrap()
            .unwrap();
        f.release.add_permits(1);
        let (socket, reply) = tokio::select! {
            reply = receive(&mut a) => ("original", reply),
            reply = receive(&mut b) => ("unrelated", reply),
        };
        assert_eq!(
            socket, "original",
            "rejected header redirected pending reply: {malformed:?}"
        );
        assert_eq!(reply.0, MessageType::Response);
        assert_eq!(reply.2.as_ref(), b"private-a");
        assert_eq!(&reply.1[..16], &ID.to_le_bytes());
        f.finish().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_worker_and_response_ids_keep_each_requests_connection() {
    for second_id in [ID + (1 << 64), ID] {
        let f = Fixture::new().await;
        let mut a = f.connect().await;
        let mut b = f.connect().await;
        f.pending(&mut a, ID, b"private-a").await;
        f.pending(&mut b, second_id, b"private-b").await;
        f.release.add_permits(2);
        let (ra, rb) = tokio::join!(receive(&mut a), receive(&mut b));
        assert_eq!(ra.2.as_ref(), b"private-a");
        assert_eq!(rb.2.as_ref(), b"private-b");
        assert_eq!(&ra.1[..16], &ID.to_le_bytes());
        assert_eq!(&rb.1[..16], &second_id.to_le_bytes());
        f.finish().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fire_and_forget_cannot_redirect_a_pending_reply() {
    let f = Fixture::new().await;
    let mut a = f.connect().await;
    let mut b = f.connect().await;
    f.pending(&mut a, ID, b"private-a").await;
    send(&mut b, &header(ID, 0, b"barrier", b""), b"").await;
    tokio::time::timeout(Duration::from_secs(2), f.barrier.recv_async())
        .await
        .unwrap()
        .unwrap();
    f.release.add_permits(1);
    let reply = receive(&mut a).await;
    assert_eq!(reply.2.as_ref(), b"private-a");
    f.finish().await;
}
