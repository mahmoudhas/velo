// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! High-performance TCP listener for ActiveMessage transport
//!
//! This module provides a TCP server that accepts incoming connections,
//! decodes framed messages using zero-copy techniques, and routes them
//! to the appropriate transport streams.

use anyhow::{Context, Result};
use futures::StreamExt;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener as TokioTcpListener, TcpStream};
use tokio::runtime::{Handle, Runtime};
use tokio_util::codec::Framed;
use tracing::{debug, error, info, warn};

use crate::observability::TransportRejection;
use crate::transports::ingress::{Routed, route_frame};

use velo_ext::{MessageType, ShutdownState, TransportAdapter, TransportErrorHandler};

use super::framing::{TcpFrameCodec, maybe_shrink_read_buffer, parse_shrink_threshold};

#[cfg(test)]
tokio::task_local! {
    /// Counts how many times a connection handler has built its teardown
    /// future. The teardown token is node-wide, so every build registers a
    /// waiter on one `Notify` that all connections share, and every drop
    /// removes it again under that `Notify`'s lock.
    static TEARDOWN_ARMS: std::sync::Arc<std::sync::atomic::AtomicU64>;
}

#[cfg(test)]
fn note_teardown_arm() {
    let _ = TEARDOWN_ARMS.try_with(|n| n.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
}

/// Compiles away: the seam must cost the shipped listener nothing.
#[cfg(not(test))]
#[inline(always)]
fn note_teardown_arm() {}

/// The one place a connection builds its teardown future, so the count the
/// test reads is the count of builds, not of a separate call that could drift
/// from them.
///
/// On a child token, not the node-wide one. The loop polls this future on
/// every frame, and each poll locks the token's tree node and, while waiting,
/// its `Notify` waiter list. On the shared token both are shared by every
/// connection on the node; a child token has its own of each, so the per-frame
/// locks are uncontended. Cancelling the node-wide token still reaches every
/// child.
fn connection_teardown(
    shutdown_state: &ShutdownState,
) -> tokio_util::sync::WaitForCancellationFutureOwned {
    note_teardown_arm();
    shutdown_state
        .teardown_token()
        .child_token()
        .cancelled_owned()
}

/// Per-connection configuration handed to [`TcpListener::handle_connection`].
struct ConnectionContext {
    adapter: TransportAdapter,
    error_handler: Arc<dyn TransportErrorHandler>,
    shutdown_state: ShutdownState,
    transport_key: String,
    metrics: Option<std::sync::Arc<dyn velo_ext::TransportObservability>>,
    shrink_threshold: usize,
}

/// One fallback reply. Dropping a queued/current frame on connection failure
/// must report it even if its admission was already acknowledged.
struct ReplyFrame {
    kind: MessageType,
    header: bytes::Bytes,
    payload: bytes::Bytes,
    on_error: Arc<dyn TransportErrorHandler>,
    runtime: Handle,
    queued_at: Option<std::time::Instant>,
    completed: bool,
    failure: Option<String>,
}

impl Drop for ReplyFrame {
    fn drop(&mut self) {
        if !self.completed {
            let on_error = self.on_error.clone();
            let header = self.header.clone();
            let payload = self.payload.clone();
            let error = self
                .failure
                .take()
                .unwrap_or_else(|| "TCP reply connection closed before write".into());
            // A gate can drop a failed frame while holding its queue lock.
            // Never invoke arbitrary callbacks under that lock: an error
            // handler is allowed to submit another send.
            drop(self.runtime.spawn(async move {
                on_error.on_error(header, payload, error);
            }));
        }
    }
}

/// Runtime configuration for the TCP listener
pub enum RuntimeConfig {
    /// Use an existing tokio runtime handle.
    Handle(Handle),
    /// Use a provided tokio runtime.
    Runtime(Arc<Runtime>),
    /// Create a single-threaded runtime pinned to the specified CPU core (Linux only).
    CpuPin(usize),
}

/// High-performance TCP listener for ActiveMessage transport
///
/// This listener accepts incoming TCP connections and routes decoded frames
/// to the appropriate transport streams with zero-copy performance.
pub struct TcpListener {
    bind_addr: SocketAddr,
    adapter: TransportAdapter,
    error_handler: Arc<dyn TransportErrorHandler>,
    shutdown_state: ShutdownState,
    runtime_config: RuntimeConfig,
    listener: Option<std::net::TcpListener>,
    transport_key: String,
    metrics: Option<std::sync::Arc<dyn velo_ext::TransportObservability>>,
    /// If the per-connection read buffer grows past this size, reset it back to
    /// [`SHRINK_RESET_CAPACITY`] once it next empties. Bounds long-lived memory
    /// after a single oversized frame.
    shrink_threshold: usize,
    /// `SO_RCVBUF`/`SO_SNDBUF` for the listening socket, or `None` for the
    /// kernel's autotuning. See [`size_socket_buffers`].
    socket_buffers: Option<usize>,
}

impl TcpListener {
    /// Create a new builder for TcpListener
    pub fn builder() -> TcpListenerBuilder {
        TcpListenerBuilder::new()
    }

    /// Start the listener and serve incoming connections
    ///
    /// This method blocks or spawns based on the runtime configuration:
    /// - For Handle/Runtime: spawns tasks and returns immediately
    /// - For CpuPin: creates a pinned runtime and blocks until cancellation
    pub async fn serve(mut self) -> Result<()> {
        // Extract runtime config to avoid borrow issues
        let runtime_config = std::mem::replace(
            &mut self.runtime_config,
            RuntimeConfig::Handle(Handle::current()),
        );

        match runtime_config {
            RuntimeConfig::Handle(handle) => {
                handle.spawn(async move {
                    if let Err(e) = self.run_server().await {
                        error!("TCP listener error: {}", e);
                    }
                });
                Ok(())
            }
            RuntimeConfig::Runtime(rt) => {
                rt.spawn(async move {
                    if let Err(e) = self.run_server().await {
                        error!("TCP listener error: {}", e);
                    }
                });
                Ok(())
            }
            RuntimeConfig::CpuPin(cpu_id) => {
                let rt = Self::create_pinned_runtime(cpu_id)
                    .context("Failed to create CPU-pinned runtime")?;
                rt.block_on(self.run_server())
            }
        }
    }

    /// Create a single-threaded runtime pinned to a specific CPU core
    #[cfg(target_os = "linux")]
    fn create_pinned_runtime(cpu_id: usize) -> Result<Runtime> {
        use nix::sched::{CpuSet, sched_setaffinity};
        use nix::unistd::Pid;

        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .thread_name("tcp-listener-pinned")
            .on_thread_start(move || {
                let mut cpu_set = CpuSet::new();
                if cpu_set.set(cpu_id).is_ok() {
                    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpu_set) {
                        error!("Failed to pin thread to CPU {}: {}", cpu_id, e);
                    } else {
                        debug!("Successfully pinned TCP listener to CPU {}", cpu_id);
                    }
                }
            })
            .build()
            .context("Failed to build tokio runtime")
    }

    /// Create a single-threaded runtime without CPU pinning (non-Linux platforms)
    #[cfg(not(target_os = "linux"))]
    fn create_pinned_runtime(cpu_id: usize) -> Result<Runtime> {
        warn!(
            "CPU pinning requested (CPU {}) but not supported on this platform",
            cpu_id
        );
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .thread_name("tcp-listener")
            .build()
            .context("Failed to build tokio runtime")
    }

    /// Main server loop that accepts connections
    async fn run_server(self) -> Result<()> {
        // Use pre-bound listener if provided, otherwise bind to the address
        let listener = if let Some(std_listener) = self.listener {
            // Best effort for caller-provided listeners: they are already
            // live, so connections that completed their handshake before this
            // point keep kernel-default autotuned buffers (safe). Listeners
            // the transport builds itself are sized before listen() instead —
            // see `size_socket_buffers`.
            size_socket_buffers(&std_listener, self.socket_buffers);

            // Set non-blocking for tokio conversion
            std_listener
                .set_nonblocking(true)
                .context("Failed to set listener to non-blocking")?;

            TokioTcpListener::from_std(std_listener)
                .context("Failed to convert std TcpListener to tokio TcpListener")?
        } else {
            let socket = match self.bind_addr {
                SocketAddr::V4(_) => tokio::net::TcpSocket::new_v4(),
                SocketAddr::V6(_) => tokio::net::TcpSocket::new_v6(),
            }
            .context("Failed to create TCP socket")?;
            socket
                .set_reuseaddr(true)
                .context("Failed to set SO_REUSEADDR")?;
            // Before listen(), so every accepted socket inherits the sizes.
            size_socket_buffers(&socket, self.socket_buffers);
            socket
                .bind(self.bind_addr)
                .context(format!("Failed to bind TCP listener to {}", self.bind_addr))?;
            // 1024 matches tokio's TcpListener::bind backlog.
            socket.listen(1024).context("Failed to listen")?
        };

        let local_addr = listener
            .local_addr()
            .context("Failed to get local address")?;
        info!("TCP listener bound to {}", local_addr);

        let teardown_token = self.shutdown_state.teardown_token().clone();

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, peer_addr)) => {
                            debug!("Accepted TCP connection from {}", peer_addr);

                            let ctx = ConnectionContext {
                                adapter: self.adapter.clone(),
                                error_handler: self.error_handler.clone(),
                                shutdown_state: self.shutdown_state.clone(),
                                transport_key: self.transport_key.clone(),
                                metrics: self.metrics.clone(),
                                shrink_threshold: self.shrink_threshold,
                            };

                            tokio::spawn(async move {
                                if let Err(e) = Self::handle_connection(stream, peer_addr, ctx).await
                                {
                                    warn!("Error handling connection from {}: {}", peer_addr, e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("Failed to accept TCP connection: {}", e);
                        }
                    }
                }
                _ = teardown_token.cancelled() => {
                    info!("TCP listener shutting down (teardown)");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a single TCP connection
    async fn handle_connection(
        stream: TcpStream,
        peer_addr: SocketAddr,
        ctx: ConnectionContext,
    ) -> Result<()> {
        let ConnectionContext {
            adapter,
            error_handler,
            shutdown_state,
            transport_key,
            metrics,
            shrink_threshold,
        } = ctx;
        debug!("Configuring connection from {}", peer_addr);

        // Configure socket for high performance
        if let Err(e) = stream.set_nodelay(true) {
            warn!("Failed to set TCP_NODELAY on {}: {}", peer_addr, e);
        }

        #[allow(deprecated)] // Intentional: linger ensures clean socket shutdown
        if let Err(e) = stream.set_linger(Some(Duration::from_secs(1))) {
            warn!("Failed to set linger on {}: {}", peer_addr, e);
        }

        // Set keep-alive to detect dead connections
        let keepalive = socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));

        let sock_ref = socket2::SockRef::from(&stream);
        if let Err(e) = sock_ref.set_tcp_keepalive(&keepalive) {
            warn!("Failed to set TCP keepalive on {}: {}", peer_addr, e);
        }

        // Legacy senders can reuse a cached _hello after the receiver restarts.
        // Preserve a connection-local response route without inventing peer
        // identity/address registration or changing the wire format.
        let (reply_tx, reply_rx) = flume::bounded::<ReplyFrame>(64);
        let reply_gate = velo_ext::AdmissionGate::new(reply_tx, Handle::current());
        let gate = reply_gate.clone();
        // Preserve InboundMessage's unwind-safe capability contract. Observer
        // hooks are diagnostic only: they cannot mutate route selection or the
        // admission gate's invariants, and a panic abandons this invocation.
        let reply_metrics = std::panic::AssertUnwindSafe(metrics.clone());
        let reply_runtime = Handle::current();
        let route: Arc<velo_ext::transport::ReplyRoute> =
            Arc::new(move |kind, header, payload, on_error| {
                let frame_bytes = header.len() + payload.len();
                let result = gate.send(ReplyFrame {
                    kind,
                    header,
                    payload,
                    on_error,
                    runtime: reply_runtime.clone(),
                    queued_at: reply_metrics.as_ref().map(|_| std::time::Instant::now()),
                    completed: false,
                    failure: None,
                });
                let Some(metrics) = reply_metrics.as_ref() else {
                    return result;
                };
                match result {
                    velo_ext::SendOutcome::Pending(admission) => {
                        metrics.record_send_backpressure();
                        let metrics = metrics.clone();
                        velo_ext::SendOutcome::Pending(admission.on_resolved(move |result| {
                            if result.is_ok() {
                                metrics.record_frame(
                                    velo_ext::Direction::Outbound,
                                    crate::transports::message_type_label(kind),
                                    frame_bytes,
                                );
                            }
                        }))
                    }
                    admitted => {
                        metrics.record_frame(
                            velo_ext::Direction::Outbound,
                            crate::transports::message_type_label(kind),
                            frame_bytes,
                        );
                        admitted
                    }
                }
            });
        let adapter = adapter.with_reply_route(route);

        // Create framed stream with zero-copy codec
        let mut framed = Framed::new(stream, TcpFrameCodec::new());
        // Built once per connection: the token is node-wide, so a future
        // rebuilt per frame would register and remove a waiter on one
        // `Notify` shared by every connection, under its lock, per frame.
        let teardown = connection_teardown(&shutdown_state);
        tokio::pin!(teardown);

        debug!("Connection from {} ready for frames", peer_addr);

        loop {
            tokio::select! {
                // Prioritize teardown so a saturated connection cannot starve shutdown.
                biased;
                _ = &mut teardown => {
                    debug!("Connection handler for {} torn down", peer_addr);
                    break;
                }
                reply = reply_rx.recv_async() => {
                    let Ok(mut reply) = reply else { break; };
                    if let (Some(metrics), Some(queued_at)) = (metrics.as_ref(), reply.queued_at) {
                        metrics.record_egress_queue_wait(queued_at.elapsed());
                    }
                    let write_started = std::time::Instant::now();
                    let written = tokio::select! {
                        biased;
                        _ = &mut teardown => break,
                        result = TcpFrameCodec::encode_frame(framed.get_mut(), reply.kind, &reply.header, &reply.payload) => result,
                    };
                    if let Some(metrics) = metrics.as_ref() {
                        metrics.record_egress_write_duration(write_started.elapsed());
                    }
                    if let Err(error) = written {
                        reply.failure = Some(error.to_string());
                        break;
                    }
                    reply.completed = true;
                    if let Some(metrics) = metrics.as_ref() {
                        metrics.record_frames_written(reply.kind, 1);
                    }
                }
                frame_result = framed.next() => {
                    match frame_result {
                        Some(Ok((msg_type, header, payload))) => {
                            let frame_size = header.len() + payload.len();
                            // Route frame to appropriate stream based on type.
                            // `route_frame` owns the drain gate for Message
                            // frames — it acquires the in-flight guard before
                            // it re-reads the flag, so an admitted message is
                            // counted work even while it is only queued.
                            // Response/Ack/Event frames always pass through.
                            match route_frame(
                                msg_type,
                                header,
                                payload,
                                &adapter,
                                &error_handler,
                                &transport_key,
                                metrics.as_ref(),
                            )
                            .await
                            {
                                Ok(Routed::Delivered) => {}
                                Ok(Routed::DrainRejected { header }) => {
                                    debug!(
                                        "Rejecting Message frame from {} during drain (sending ShuttingDown)",
                                        peer_addr
                                    );
                                    // Echo original header back for correlation, empty payload
                                    if let Err(e) = TcpFrameCodec::encode_frame(
                                        framed.get_mut(),
                                        MessageType::ShuttingDown,
                                        &header,
                                        &[],
                                    )
                                    .await
                                    {
                                        warn!(
                                            "Failed to send ShuttingDown frame to {}: {}",
                                            peer_addr, e
                                        );
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to route {:?} frame from {}: {}",
                                        msg_type, peer_addr, e
                                    );
                                }
                            }

                            maybe_shrink_read_buffer(framed.read_buffer_mut(), shrink_threshold, frame_size);
                        }
                        Some(Err(e)) => {
                            if let Some(metrics) = metrics.as_ref() {
                                metrics.record_rejection(TransportRejection::DecodeError);
                            }
                            error!("Frame decode error from {}: {}", peer_addr, e);
                            break;
                        }
                        None => {
                            // Connection closed gracefully (FIN received)
                            debug!("Connection from {} closed gracefully", peer_addr);
                            break;
                        }
                    }
                }
            }
        }

        reply_gate.fail_all(velo_ext::AdmissionError::ChannelClosed);
        while reply_rx.try_recv().is_ok() {}
        Ok(())
    }
}

/// Builder for TcpListener
pub struct TcpListenerBuilder {
    bind_addr: Option<SocketAddr>,
    adapter: Option<TransportAdapter>,
    error_handler: Option<Arc<dyn TransportErrorHandler>>,
    shutdown_state: Option<ShutdownState>,
    runtime_config: Option<RuntimeConfig>,
    listener: Option<std::net::TcpListener>,
    transport_key: Option<String>,
    metrics: Option<std::sync::Arc<dyn velo_ext::TransportObservability>>,
    shrink_threshold: Option<usize>,
    socket_buffers: Option<usize>,
}

impl TcpListenerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            bind_addr: None,
            adapter: None,
            error_handler: None,
            shutdown_state: None,
            runtime_config: None,
            listener: None,
            transport_key: None,
            metrics: None,
            shrink_threshold: None,
            socket_buffers: Some(DEFAULT_SOCKET_BUFFERS),
        }
    }

    /// Set the bind address
    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }

    /// Set the transport adapter
    pub fn adapter(mut self, adapter: TransportAdapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Set the error handler
    pub fn error_handler(mut self, handler: Arc<dyn TransportErrorHandler>) -> Self {
        self.error_handler = Some(handler);
        self
    }

    /// Set the shutdown state for graceful drain coordination
    pub fn shutdown_state(mut self, state: ShutdownState) -> Self {
        self.shutdown_state = Some(state);
        self
    }

    /// Use an existing tokio runtime handle
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.runtime_config = Some(RuntimeConfig::Handle(handle));
        self
    }

    /// Use a provided tokio runtime
    pub fn with_runtime(mut self, runtime: Arc<Runtime>) -> Self {
        self.runtime_config = Some(RuntimeConfig::Runtime(runtime));
        self
    }

    /// Create a single-threaded runtime pinned to a specific CPU core
    pub fn with_cpu_pin(mut self, cpu_id: usize) -> Self {
        self.runtime_config = Some(RuntimeConfig::CpuPin(cpu_id));
        self
    }

    /// Use a pre-bound TcpListener
    ///
    /// This is useful for tests where you want to bind to port 0 and avoid port races.
    /// When provided, the bind_addr should still be set (for logging/debugging purposes).
    pub fn listener(mut self, listener: Option<std::net::TcpListener>) -> Self {
        self.listener = listener;
        self
    }

    /// Set the transport key used for metrics labels.
    pub fn transport_key(mut self, transport_key: impl Into<String>) -> Self {
        self.transport_key = Some(transport_key.into());
        self
    }

    /// Install transport-scoped observability handles.
    pub fn metrics(
        mut self,
        metrics: Option<std::sync::Arc<dyn velo_ext::TransportObservability>>,
    ) -> Self {
        self.metrics = metrics;
        self
    }

    /// Override the read-buffer shrink threshold (default 8 MB).
    ///
    /// When a per-connection read buffer's capacity grows past this many bytes,
    /// it will be reset back to a small capacity the next time it drains. This
    /// bounds long-lived memory after a single oversized frame.
    pub fn shrink_threshold(mut self, bytes: usize) -> Self {
        self.shrink_threshold = Some(bytes);
        self
    }

    /// Socket buffer size for accepted connections, or `None` for the
    /// kernel's autotuning (default: [`DEFAULT_SOCKET_BUFFERS`]). See
    /// `TcpTransportBuilder::socket_buffers` for the trade-off.
    pub fn socket_buffers(mut self, bytes: Option<usize>) -> Self {
        self.socket_buffers = bytes;
        self
    }

    /// Build the TcpListener
    pub fn build(self) -> Result<TcpListener> {
        let bind_addr = self
            .bind_addr
            .ok_or_else(|| anyhow::anyhow!("bind_addr is required"))?;
        let adapter = self
            .adapter
            .ok_or_else(|| anyhow::anyhow!("adapter is required"))?;
        let error_handler = self
            .error_handler
            .ok_or_else(|| anyhow::anyhow!("error_handler is required"))?;
        let shutdown_state = self.shutdown_state.unwrap_or_default();
        let runtime_config = self
            .runtime_config
            .unwrap_or_else(|| RuntimeConfig::Handle(Handle::current()));

        Ok(TcpListener {
            bind_addr,
            adapter,
            error_handler,
            shutdown_state,
            runtime_config,
            listener: self.listener,
            transport_key: self.transport_key.unwrap_or_else(|| "tcp".to_string()),
            metrics: self.metrics,
            shrink_threshold: self
                .shrink_threshold
                .unwrap_or_else(default_shrink_threshold),
            socket_buffers: self.socket_buffers,
        })
    }
}

/// Resolve the default read-buffer shrink threshold for the TCP listener
/// from `VELO_TCP_SHRINK_THRESHOLD` (in bytes), falling back to the codec-
/// level default.
pub(super) fn default_shrink_threshold() -> usize {
    parse_shrink_threshold(std::env::var("VELO_TCP_SHRINK_THRESHOLD").ok().as_deref())
}

/// Size a socket's buffers: a listening socket before `listen()`, a dialed
/// socket before its first write, and never an accepted socket.
///
/// The kernel snapshots the listener's buffer sizes into each child socket
/// when it creates the child during the TCP handshake, so sizing the listener
/// before `listen()` guarantees every accepted socket carries the values
/// before the peer's first data byte can arrive. Setting SO_RCVBUF on an
/// accepted socket instead races the peer's initial burst: velo connections
/// are unidirectional and the dialer starts writing the instant `connect()`
/// returns, while the accept handler runs a task-spawn later. On Linux the
/// late setsockopt locks the buffer (disabling receive autotuning) and,
/// applied after in-flight data has already grown the window, permanently
/// collapses the advertised window to a few KB (observed: 32 KB stuck window,
/// 100% rwnd-limited, ~100x throughput loss on loopback where rmem_max clamps
/// the 2 MB request to ~416 KB).
///
/// Best effort: failure to size is logged, never fatal — kernel-default
/// autotuned buffers are safe. `None` leaves the buffers to the kernel.
pub(super) fn size_socket_buffers<'a>(sock: impl Into<socket2::SockRef<'a>>, bytes: Option<usize>) {
    let Some(bytes) = bytes else {
        return;
    };
    let sock = sock.into();
    if let Err(e) = sock.set_recv_buffer_size(bytes) {
        warn!("Failed to set socket recv buffer size: {}", e);
    }
    if let Err(e) = sock.set_send_buffer_size(bytes) {
        warn!("Failed to set socket send buffer size: {}", e);
    }
}

/// The socket buffer size TCP sets unless a caller chooses otherwise.
pub const DEFAULT_SOCKET_BUFFERS: usize = 2 * 1024 * 1024;

impl Default for TcpListenerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transports::transport::make_channels;
    use bytes::Bytes;
    use std::net::{IpAddr, Ipv4Addr};

    struct TestErrorHandler;

    impl TransportErrorHandler for TestErrorHandler {
        fn on_error(&self, _header: Bytes, _payload: Bytes, error: String) {
            eprintln!("Test error handler: {}", error);
        }
    }

    struct ReenteringReplyError {
        gate: std::sync::OnceLock<velo_ext::AdmissionGate<ReplyFrame>>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ReenteringReplyError {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                gate: std::sync::OnceLock::new(),
                calls: std::sync::atomic::AtomicUsize::new(0),
            })
        }
        async fn wait_for(&self, count: usize) {
            tokio::time::timeout(Duration::from_secs(2), async {
                while self.calls.load(std::sync::atomic::Ordering::SeqCst) != count {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("reply failure callbacks did not complete");
        }
    }

    impl TransportErrorHandler for ReenteringReplyError {
        fn on_error(&self, _: Bytes, _: Bytes, _: String) {
            // Takes the gate's queue lock: an inline Drop callback deadlocks.
            let _ = self.gate.get().unwrap().queued_len();
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn reply_frame(errors: Arc<dyn TransportErrorHandler>) -> ReplyFrame {
        ReplyFrame {
            kind: MessageType::Response,
            header: Bytes::from_static(b"header"),
            payload: Bytes::from_static(b"payload"),
            on_error: errors,
            runtime: Handle::current(),
            queued_at: None,
            completed: false,
            failure: None,
        }
    }

    #[tokio::test]
    async fn closed_reply_gate_reports_failure_without_calling_under_its_lock() {
        let (tx, rx) = flume::bounded(1);
        let gate = velo_ext::AdmissionGate::new(tx, Handle::current());
        let errors = ReenteringReplyError::new();
        errors.gate.set(gate.clone()).unwrap();
        drop(rx);
        let admission = gate
            .send(reply_frame(errors.clone()))
            .into_pending()
            .unwrap();
        assert_eq!(
            admission.await,
            Err(velo_ext::AdmissionError::ChannelClosed)
        );
        errors.wait_for(1).await;
    }

    #[tokio::test]
    async fn pending_reply_cancellation_does_not_double_report_or_drop_the_admitted_reply() {
        let (tx, rx) = flume::bounded(1);
        let gate = velo_ext::AdmissionGate::new(tx, Handle::current());
        let errors = ReenteringReplyError::new();
        errors.gate.set(gate.clone()).unwrap();
        assert!(gate.send(reply_frame(errors.clone())).is_admitted());
        let pending = gate
            .send(reply_frame(errors.clone()))
            .into_pending()
            .unwrap();
        pending.cancel();
        errors.wait_for(1).await;
        let mut written = rx.try_recv().unwrap();
        written.completed = true;
        drop(written);
        assert!(rx.try_recv().is_err());
        assert_eq!(errors.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn reply_gate_teardown_reports_each_queued_frame_once() {
        let (tx, rx) = flume::bounded(1);
        let gate = velo_ext::AdmissionGate::new(tx, Handle::current());
        let errors = ReenteringReplyError::new();
        errors.gate.set(gate.clone()).unwrap();
        assert!(gate.send(reply_frame(errors.clone())).is_admitted());
        let pending: Vec<_> = (0..32)
            .map(|_| {
                gate.send(reply_frame(errors.clone()))
                    .into_pending()
                    .unwrap()
            })
            .collect();
        gate.fail_all(velo_ext::AdmissionError::ChannelClosed);
        for admission in pending {
            assert_eq!(
                admission.await,
                Err(velo_ext::AdmissionError::ChannelClosed)
            );
        }
        drop(rx.try_recv().unwrap());
        errors.wait_for(33).await;
        assert!(rx.try_recv().is_err());
    }

    /// A connection builds its teardown future once, not once per frame.
    ///
    /// The token is node-wide, so a future rebuilt per frame registers and
    /// removes a waiter on one `Notify` shared by every connection -- about 5
    /// percent of a frontend's CPU at small batches. Teardown must still end
    /// the loop while frames are flowing, which is what the rebuilt future
    /// bought, so the client here never stops sending: the cancel lands in the
    /// middle of the stream.
    #[tokio::test]
    async fn a_connection_arms_its_teardown_once() {
        let listener = TokioTcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = tokio::spawn(async move {
            let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
            // Until the listener side closes the connection.
            while TcpFrameCodec::encode_frame(&mut s, MessageType::Response, b"h", b"p")
                .await
                .is_ok()
            {}
        });
        let (stream, peer) = listener.accept().await.unwrap();
        let (adapter, streams) = make_channels();
        let shutdown_state = ShutdownState::new();
        let ctx = ConnectionContext {
            adapter,
            error_handler: Arc::new(TestErrorHandler),
            shutdown_state: shutdown_state.clone(),
            transport_key: "tcp".to_string(),
            metrics: None,
            shrink_threshold: usize::MAX,
        };
        let arms = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let handler = tokio::spawn(TEARDOWN_ARMS.scope(
            arms.clone(),
            TcpListener::handle_connection(stream, peer, ctx),
        ));
        for _ in 0..50 {
            tokio::time::timeout(Duration::from_secs(5), streams.response_stream.recv_async())
                .await
                .expect("frame routed")
                .unwrap();
        }
        // Keep taking what is routed, so a full channel is not what stops the
        // loop; only teardown should.
        let responses = streams.response_stream.clone();
        let drainer = tokio::spawn(async move { while responses.recv_async().await.is_ok() {} });
        assert!(!client.is_finished(), "the client must still be sending");
        shutdown_state.teardown_token().cancel();
        tokio::time::timeout(Duration::from_secs(5), handler)
            .await
            .expect("teardown ends the connection loop while frames are flowing")
            .unwrap()
            .unwrap();
        assert_eq!(arms.load(std::sync::atomic::Ordering::Relaxed), 1);
        drop(streams);
        client.abort();
        drainer.abort();
    }

    #[test]
    fn test_builder_requires_fields() {
        let result = TcpListener::builder().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_builder_with_all_fields() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (adapter, _streams) = make_channels();
        let error_handler = Arc::new(TestErrorHandler);

        let result = TcpListener::builder()
            .bind_addr(bind_addr)
            .adapter(adapter)
            .error_handler(error_handler)
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_with_cpu_pin() {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let (adapter, _streams) = make_channels();
        let error_handler = Arc::new(TestErrorHandler);

        let result = TcpListener::builder()
            .bind_addr(bind_addr)
            .adapter(adapter)
            .error_handler(error_handler)
            .with_cpu_pin(0)
            .build();

        assert!(result.is_ok());
    }
}
