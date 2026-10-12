// SPDX-FileCopyrightText: Copyright (c) 2026 Velo contributors.
// SPDX-License-Identifier: Apache-2.0

//! A surviving caller retains its _hello cache when a different receiver
//! incarnation binds the same address. The receiver must be able to answer
//! without registering a made-up identity for that caller. All traffic here
//! is synthetic and stays on loopback; no RDMA device or broker is required.
use std::{net::SocketAddr, sync::Arc, time::Duration};

use bytes::Bytes;
use velo::messenger::Messenger;
use velo::transports::{ShutdownPolicy, tcp::TcpTransportBuilder};
use velo::{Handler, InstanceId, TypedContext};

async fn node(address: Option<SocketAddr>) -> (Arc<Messenger>, SocketAddr) {
    let first_incarnation = address.is_none();
    let address = address.unwrap_or_else(|| "127.0.0.1:0".parse().unwrap());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    // The pinned version's shutdown may finish before its accept loop releases
    // the listener. Do not confuse that asynchronous close with the reply bug.
    let listener = loop {
        match std::net::TcpListener::bind(address) {
            Ok(listener) => break listener,
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "listener did not close"
                );
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            Err(error) => panic!("bind: {error}"),
        }
    };
    let address = listener.local_addr().unwrap();
    let tcp = TcpTransportBuilder::new()
        .from_listener(listener)
        .unwrap()
        .build()
        .unwrap();
    let messenger = Messenger::builder()
        .add_transport(Arc::new(tcp))
        .build()
        .await
        .unwrap();
    messenger
        .register_handler(Handler::unary_handler("echo", |ctx| Ok(Some(ctx.payload))).build())
        .unwrap();
    messenger
        .register_handler(
            Handler::typed_unary("typed", |ctx: TypedContext<u64>| Ok(ctx.input)).build(),
        )
        .unwrap();
    messenger
        .register_handler(
            Handler::unary_handler("fail", |_| anyhow::bail!("synthetic handler error")).build(),
        )
        .unwrap();
    messenger
        .register_handler(
            Handler::unary_handler("panic", |_| panic!("synthetic handler panic")).build(),
        )
        .unwrap();
    if first_incarnation {
        messenger
            .register_handler(
                Handler::unary_handler("removed", |ctx| Ok(Some(ctx.payload))).build(),
            )
            .unwrap();
    }
    (messenger, address)
}

async fn echo(client: &Messenger, target: InstanceId, payload: Bytes) -> anyhow::Result<Bytes> {
    tokio::time::timeout(
        Duration::from_secs(2),
        client
            .unary("echo")?
            .raw_payload(payload)
            .instance(target)
            .send(),
    )
    .await?
}

async fn warmed_pair() -> (Arc<Messenger>, Arc<Messenger>, SocketAddr) {
    let (receiver, address) = node(None).await;
    let (caller, _) = node(None).await;
    caller.register_peer(receiver.peer_info()).unwrap();
    let payload = Bytes::from_static(b"warm hello cache");
    assert_eq!(
        echo(&caller, receiver.instance_id(), payload.clone())
            .await
            .unwrap(),
        payload
    );
    (caller, receiver, address)
}

/// Before the fix, the receiver logs WorkerNotRegistered while every call waits
/// for the caller's deadline. In particular, retrying the same cached target
/// does not heal it. No refresh/prewarm step is allowed after the restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_caller_gets_replies_after_same_address_restart() {
    let (caller, receiver, address) = warmed_pair().await;
    let retired = receiver.instance_id();
    receiver
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    let (replacement, _) = node(Some(address)).await;
    assert_ne!(retired, replacement.instance_id());
    for i in 0..3u8 {
        let payload = Bytes::from(vec![i; 256]);
        assert_eq!(
            echo(&caller, retired, payload.clone()).await.unwrap(),
            payload
        );
    }
    let typed: u64 = tokio::time::timeout(
        Duration::from_secs(2),
        caller
            .typed_unary("typed")
            .unwrap()
            .payload(42u64)
            .unwrap()
            .instance(retired)
            .send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(typed, 42);
    // The current incarnation still supports the normal handshake/peer route.
    caller.register_peer(replacement.peer_info()).unwrap();
    assert_eq!(
        echo(
            &caller,
            replacement.instance_id(),
            Bytes::from_static(b"current")
        )
        .await
        .unwrap(),
        "current"
    );
    replacement
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    caller.graceful_shutdown(ShutdownPolicy::WaitForever).await;
}

/// Correlation is owned by the caller's response manager, not a handshake with
/// the restarted receiver. Concurrent, differently sized replies must not cross.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_caller_correlates_concurrent_replies() {
    let (caller, receiver, address) = warmed_pair().await;
    let retired = receiver.instance_id();
    receiver
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    let (replacement, _) = node(Some(address)).await;
    let calls = (0..128u8).map(|i| {
        let caller = caller.clone();
        async move {
            let payload = Bytes::from(vec![i; 32768 + i as usize]);
            assert_eq!(
                echo(&caller, retired, payload.clone()).await.unwrap(),
                payload
            );
        }
    });
    tokio::time::timeout(Duration::from_secs(10), futures::future::join_all(calls))
        .await
        .unwrap();
    replacement
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    caller.graceful_shutdown(ShutdownPolicy::WaitForever).await;
}

/// Error and panic replies use the same return route. Missing-handler replies
/// matter during startup, before the application registers its own handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_caller_gets_negative_replies_without_a_timeout() {
    let (caller, receiver, address) = warmed_pair().await;
    let retired = receiver.instance_id();
    receiver
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    let (replacement, _) = node(Some(address)).await;
    for (handler, expected) in [
        ("fail", "synthetic handler error"),
        ("panic", "handler panicked"),
        ("removed", "not found"),
    ] {
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            caller.unary(handler).unwrap().instance(retired).send(),
        )
        .await
        .expect("reply timed out")
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{handler}: {error:#}");
    }
    replacement
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    caller.graceful_shutdown(ShutdownPolicy::WaitForever).await;
}

/// A reply route must not keep an old socket alive, nor be reused as the route
/// for a different receiver generation. The caller deliberately keeps its
/// original target identity and cached handler list through all three restarts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_restarts_do_not_retain_dead_reply_connections() {
    let (caller, mut receiver, address) = warmed_pair().await;
    let retired = receiver.instance_id();
    for i in 0..3u8 {
        receiver
            .graceful_shutdown(ShutdownPolicy::WaitForever)
            .await;
        receiver = node(Some(address)).await.0;
        assert_eq!(
            echo(&caller, retired, Bytes::from(vec![i])).await.unwrap(),
            Bytes::from(vec![i])
        );
    }
    receiver
        .graceful_shutdown(ShutdownPolicy::WaitForever)
        .await;
    caller.graceful_shutdown(ShutdownPolicy::WaitForever).await;
}
