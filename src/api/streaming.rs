use crate::sim::WorldSnapshot;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// A backpressure-aware buffer for streaming simulation ticks to slow consumers.
///
/// When consumers (e.g., API clients) fall behind, this buffer drops old events
/// to prevent unbounded memory growth. Clients must handle gaps and catch up.
///
/// # Backpressure Strategy
/// - When buffer reaches `max_size`, oldest events are dropped
/// - Clients that fall too far behind may miss events
/// - Event IDs include tick numbers for gap detection
#[derive(Debug)]
pub struct TickBuffer {
    /// FIFO queue of snapshots waiting to be sent
    snapshots: Arc<Mutex<VecDeque<WorldSnapshot>>>,
    /// Maximum number of snapshots to buffer before dropping old ones
    max_size: usize,
    /// Current tick counter (for monitoring)
    tick_count: Arc<Mutex<u32>>,
}

impl TickBuffer {
    /// Create a new tick buffer with specified max size.
    ///
    /// # Arguments
    /// * `max_size` - Maximum snapshots to hold; oldest are dropped when exceeded
    ///
    /// # Panics
    /// Panics if `max_size` is 0.
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0, "max_size must be >= 1");
        Self {
            snapshots: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            tick_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Push a snapshot into the buffer, dropping old events if necessary.
    ///
    /// If the buffer is at capacity, the oldest snapshot is dropped to make room.
    ///
    /// # Returns
    /// `Some(dropped_snapshot)` if a snapshot was dropped due to backpressure, otherwise `None`
    pub fn push(&self, snapshot: WorldSnapshot) -> Option<WorldSnapshot> {
        let mut snapshots = self.snapshots.lock();
        let mut dropped = None;

        // Drop oldest if at capacity
        if snapshots.len() >= self.max_size {
            dropped = snapshots.pop_front();
        }

        snapshots.push_back(snapshot);
        *self.tick_count.lock() += 1;
        dropped
    }

    /// Get all buffered snapshots without removing them.
    ///
    /// Returns a clone of all snapshots currently in the buffer.
    pub fn peek_all(&self) -> Vec<WorldSnapshot> {
        self.snapshots.lock().iter().cloned().collect()
    }

    /// Drain and return all buffered snapshots.
    ///
    /// Clears the buffer. Used by streaming consumers to fetch all available ticks.
    pub fn drain_all(&self) -> Vec<WorldSnapshot> {
        let mut snapshots = self.snapshots.lock();
        snapshots.drain(..).collect()
    }

    /// Get the number of buffered snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.lock().len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.lock().is_empty()
    }

    /// Get the total number of ticks that have passed through this buffer.
    pub fn total_ticks(&self) -> u32 {
        *self.tick_count.lock()
    }

    /// Get the maximum capacity of this buffer.
    pub fn max_capacity(&self) -> usize {
        self.max_size
    }

    /// Get backpressure statistics.
    ///
    /// Returns (current_buffered, total_ticks_seen).
    pub fn stats(&self) -> (usize, u32) {
        (self.snapshots.lock().len(), *self.tick_count.lock())
    }
}

/// Adapter to integrate SimEngine's broadcast channel with the TickBuffer.
///
/// Receives ticks from SimEngine's broadcast channel and pushes them into
/// a bounded buffer for HTTP streaming clients.
pub struct StreamAdapter {
    buffer: Arc<TickBuffer>,
}

impl StreamAdapter {
    /// Create a new stream adapter with the given buffer size.
    pub fn new(buffer_size: usize) -> Self {
        Self { buffer: Arc::new(TickBuffer::new(buffer_size)) }
    }

    /// Get a reference to the underlying buffer.
    pub fn buffer(&self) -> Arc<TickBuffer> {
        Arc::clone(&self.buffer)
    }

    /// Push a snapshot from SimEngine into the buffer.
    ///
    /// Returns `Some(dropped)` if a snapshot was dropped due to backpressure.
    pub fn push_snapshot(&self, snapshot: WorldSnapshot) -> Option<WorldSnapshot> {
        self.buffer.push(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(tick: u32) -> WorldSnapshot {
        WorldSnapshot {
            tick,
            agents: Default::default(),
            events: Vec::new(),
            variables: Default::default(),
        }
    }

    #[test]
    fn test_tick_buffer_creation() {
        let buffer = TickBuffer::new(10);
        assert_eq!(buffer.max_capacity(), 10);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_tick_buffer_push() {
        let buffer = TickBuffer::new(5);
        let snap = make_snapshot(0);
        let dropped = buffer.push(snap);

        assert!(dropped.is_none());
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.total_ticks(), 1);
    }

    #[test]
    fn test_tick_buffer_backpressure() {
        let buffer = TickBuffer::new(3);

        // Fill buffer to capacity
        for i in 0..3 {
            let dropped = buffer.push(make_snapshot(i));
            assert!(dropped.is_none());
        }

        assert_eq!(buffer.len(), 3);

        // Push another: should drop oldest
        let dropped = buffer.push(make_snapshot(3));
        assert!(dropped.is_some());
        assert_eq!(dropped.unwrap().tick, 0);

        // Buffer should still be size 3
        assert_eq!(buffer.len(), 3);

        // Verify newest snapshot is there
        let all = buffer.peek_all();
        assert_eq!(all.last().unwrap().tick, 3);
    }

    #[test]
    fn test_tick_buffer_drain() {
        let buffer = TickBuffer::new(5);

        for i in 0..3 {
            buffer.push(make_snapshot(i));
        }

        let drained = buffer.drain_all();
        assert_eq!(drained.len(), 3);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_tick_buffer_peek_preserves() {
        let buffer = TickBuffer::new(5);

        for i in 0..2 {
            buffer.push(make_snapshot(i));
        }

        let peeked = buffer.peek_all();
        assert_eq!(peeked.len(), 2);

        // Buffer should still have snapshots
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_tick_buffer_stats() {
        let buffer = TickBuffer::new(3);

        for i in 0..5 {
            buffer.push(make_snapshot(i));
        }

        let (buffered, total) = buffer.stats();
        assert_eq!(buffered, 3); // Should have last 3 (2, 3, 4)
        assert_eq!(total, 5); // Total seen = 5
    }

    #[test]
    fn test_stream_adapter_creation() {
        let adapter = StreamAdapter::new(10);
        assert_eq!(adapter.buffer().max_capacity(), 10);
    }

    #[test]
    fn test_stream_adapter_push() {
        let adapter = StreamAdapter::new(2);

        adapter.push_snapshot(make_snapshot(0));
        adapter.push_snapshot(make_snapshot(1));

        // Third push should drop first
        let dropped = adapter.push_snapshot(make_snapshot(2));
        assert!(dropped.is_some());
    }

    #[test]
    #[should_panic(expected = "max_size must be >= 1")]
    fn test_tick_buffer_zero_size_panics() {
        let _buffer = TickBuffer::new(0);
    }
}
