#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace roomie {

enum class ChannelPolicy {
  kDropOldest,
  kRejectNewest,
  kReliableBlocking,
  kLatestByKey,
  kLatest,
};

enum class PushOutcome {
  kAccepted,
  kReplaced,
  kRejected,
  kTimedOut,
  kStopped,
};

enum class ChannelReplacementReason {
  kNone,
  kCapacity,
  kMatchingKey,
  kLatest,
};

struct ChannelStats {
  ChannelPolicy policy = ChannelPolicy::kDropOldest;
  std::size_t capacity = 0;
  std::size_t depth = 0;
  std::size_t high_watermark = 0;
  bool stopped = false;

  // The push outcome counters are mutually exclusive. A replacement is an
  // accepted ingress item, but is counted in replaced rather than accepted.
  std::uint64_t accepted = 0;
  std::uint64_t replaced = 0;
  std::uint64_t rejected = 0;
  std::uint64_t timed_out = 0;
  std::uint64_t stopped_pushes = 0;
  std::uint64_t dequeued = 0;
  // Explicit lifecycle cancellation is distinct from capacity replacement.
  // barrier_shed counts ordinary, replaceable work sampled before admitting a
  // protected causal barrier.
  std::uint64_t cancelled = 0;
  std::uint64_t barrier_shed = 0;

  std::uint64_t producer_wait_count = 0;
  std::uint64_t consumer_wait_count = 0;
  std::uint64_t consumer_timeout_count = 0;
  std::chrono::nanoseconds producer_wait{0};
  std::chrono::nanoseconds consumer_wait{0};

  // oldest_age is sampled when stats() is called. The dequeue ages describe
  // the most recently dequeued item and the maximum observed queue age.
  std::chrono::nanoseconds oldest_age{0};
  std::chrono::nanoseconds last_dequeue_age{0};
  std::chrono::nanoseconds max_dequeue_age{0};
};

template <typename T>
struct PushResult {
  PushOutcome outcome = PushOutcome::kRejected;
  ChannelReplacementReason replacement_reason =
      ChannelReplacementReason::kNone;

  // A replacement returns the displaced item so the caller can log its id.
  std::optional<T> replaced_item;

  // A rejected, timed-out, or stopped push returns ownership to the caller.
  std::optional<T> unconsumed_item;

  bool accepted() const {
    return outcome == PushOutcome::kAccepted ||
           outcome == PushOutcome::kReplaced;
  }

  explicit operator bool() const { return accepted(); }
};

// A bounded, policy-aware multi-producer/multi-consumer channel.
//
// A capacity of zero intentionally means unbounded. This preserves the legacy
// ThreadSafeQueue behavior and avoids giving zero the surprising meaning of a
// permanently full channel. kLatest still retains only one item, and
// kLatestByKey still retains at most one pending item for each key.
//
// LatestByKey uses an equality predicate instead of imposing a key type on T.
// For example:
//   BoundedChannel<Request> requests(
//       4, ChannelPolicy::kLatestByKey,
//       [](const Request& lhs, const Request& rhs) {
//         return lhs.camera_id == rhs.camera_id;
//       });
template <typename T>
class BoundedChannel {
 public:
  using Clock = std::chrono::steady_clock;
  using SameKey = std::function<bool(const T&, const T&)>;

  explicit BoundedChannel(
      std::size_t capacity,
      ChannelPolicy policy = ChannelPolicy::kDropOldest,
      SameKey same_key = SameKey{})
      : capacity_(capacity), policy_(policy), same_key_(std::move(same_key)) {
    if (policy_ == ChannelPolicy::kLatestByKey && !same_key_) {
      throw std::invalid_argument(
          "LatestByKey channel requires a same-key predicate");
    }
    stats_.policy = policy_;
    stats_.capacity = capacity_;
  }

  BoundedChannel(const BoundedChannel&) = delete;
  BoundedChannel& operator=(const BoundedChannel&) = delete;

  PushResult<T> push(T item) {
    return pushWithPolicy(std::move(item), policy_, std::nullopt);
  }

  // LatestByKey with reliable cross-key admission. A matching pending item is
  // superseded immediately, while a distinct key waits for capacity instead
  // of evicting unrelated work. stop() wakes blocked producers and returns
  // their unconsumed item with kStopped.
  //
  // This is intentionally an explicit operation rather than a new channel
  // policy: most LatestByKey users (for example live camera frames) want
  // capacity replacement, while durable convergence work does not.
  PushResult<T> pushLatestByKeyReliably(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!same_key_) {
      throw std::logic_error(
          "reliable LatestByKey operation requires a same-key predicate");
    }

    bool waited = false;
    Clock::time_point wait_start;
    const auto finish_wait = [&]() {
      if (waited) {
        stats_.producer_wait +=
            elapsedNanoseconds(wait_start, Clock::now());
      }
    };
    const auto admission_ready = [&]() {
      if (stopped_) {
        return true;
      }
      const auto matching = std::find_if(
          queue_.begin(), queue_.end(), [&](const Entry& entry) {
            return same_key_(entry.item, item);
          });
      if (matching != queue_.end()) {
        return !matching->replacement_protected;
      }
      return !fullLocked();
    };

    for (;;) {
      if (stopped_) {
        finish_wait();
        return stoppedResult(std::move(item));
      }

      const Clock::time_point now = Clock::now();
      const auto matching = std::find_if(
          queue_.begin(), queue_.end(), [&](const Entry& entry) {
            return same_key_(entry.item, item);
          });
      if (matching != queue_.end() &&
          !matching->replacement_protected) {
        T replaced = std::move(matching->item);
        queue_.erase(matching);
        enqueueLocked(std::move(item), now);
        finish_wait();
        PushResult<T> result = replacedResult(
            std::move(replaced), ChannelReplacementReason::kMatchingKey);
        lock.unlock();
        not_empty_cv_.notify_one();
        return result;
      }
      if (matching == queue_.end() && !fullLocked()) {
        enqueueLocked(std::move(item), now);
        finish_wait();
        PushResult<T> result = acceptedResult();
        lock.unlock();
        not_empty_cv_.notify_one();
        return result;
      }

      if (!waited) {
        waited = true;
        wait_start = Clock::now();
        ++stats_.producer_wait_count;
      }
      not_full_cv_.wait(lock, admission_ready);
    }
  }

  // Allows admission to strengthen a single item's delivery contract without
  // creating a second queue. Roomie uses this for perception-candidate map
  // work: ordinary map frames follow DropOldest, while an admitted candidate
  // enters the same FIFO with ReliableBlocking semantics.
  PushResult<T> pushUsingPolicy(T item, ChannelPolicy policy) {
    return pushWithPolicy(std::move(item), policy, std::nullopt);
  }

  // Admit a replacement-protected causal barrier without allowing a backlog
  // of ordinary work to consume its deadline. Older replaceable entries are
  // sampled until at most max_replaceable_ahead remain; protected entries are
  // never reordered or evicted. If protected work alone fills the channel,
  // the producer waits using the supplied deadline just like ReliableBlocking.
  template <typename Rep, typename Period>
  PushResult<T> pushBarrierFor(
      T item,
      std::size_t max_replaceable_ahead,
      const std::chrono::duration<Rep, Period>& timeout) {
    Clock::duration bounded_timeout =
        std::chrono::duration_cast<Clock::duration>(timeout);
    if (bounded_timeout < Clock::duration::zero()) {
      bounded_timeout = Clock::duration::zero();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      return stoppedResult(std::move(item));
    }

    auto replaceable_count = [&]() {
      return static_cast<std::size_t>(std::count_if(
          queue_.begin(), queue_.end(), [](const Entry& entry) {
            return !entry.replacement_protected;
          }));
    };
    while (replaceable_count() > max_replaceable_ahead) {
      auto replaceable = oldestReplaceableLocked();
      if (replaceable == queue_.end()) {
        break;
      }
      queue_.erase(replaceable);
      ++stats_.barrier_shed;
    }
    if (!fullLocked()) {
      enqueueLocked(std::move(item), Clock::now(), true);
      PushResult<T> result = acceptedResult();
      lock.unlock();
      not_empty_cv_.notify_one();
      not_full_cv_.notify_all();
      return result;
    }

    return pushReliableLocked(std::move(item), bounded_timeout, &lock);
  }

  template <typename Rep, typename Period>
  PushResult<T> pushUsingPolicyFor(
      T item,
      ChannelPolicy policy,
      const std::chrono::duration<Rep, Period>& timeout) {
    Clock::duration bounded_timeout =
        std::chrono::duration_cast<Clock::duration>(timeout);
    if (bounded_timeout < Clock::duration::zero()) {
      bounded_timeout = Clock::duration::zero();
    }
    return pushWithPolicy(std::move(item), policy, bounded_timeout);
  }

  template <typename Rep, typename Period>
  PushResult<T> pushFor(
      T item, const std::chrono::duration<Rep, Period>& timeout) {
    Clock::duration bounded_timeout =
        std::chrono::duration_cast<Clock::duration>(timeout);
    if (bounded_timeout < Clock::duration::zero()) {
      bounded_timeout = Clock::duration::zero();
    }
    return pushWithPolicy(std::move(item), policy_, bounded_timeout);
  }

  // Non-blocking for every policy, including ReliableBlocking.
  PushResult<T> tryPush(T item) {
    return pushWithPolicy(
        std::move(item), policy_, Clock::duration::zero());
  }

  // Compatibility surface for existing Roomie producers. This operation
  // deliberately preserves its named legacy behavior, independent of the
  // channel's configured default policy. New code should call push().
  bool pushDropOldest(T item) {
    return pushWithPolicy(std::move(item),
                          ChannelPolicy::kDropOldest,
                          Clock::duration::zero())
        .accepted();
  }

  bool tryPop(T* item) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    popLocked(item);
    lock.unlock();
    not_full_cv_.notify_one();
    return true;
  }

  // Cancel matching queued work, including replacement-protected entries.
  // This is reserved for explicit lifecycle termination (supersede, deadline,
  // or shutdown), not ordinary capacity management.
  template <typename Predicate>
  std::size_t cancelIf(Predicate predicate) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::size_t removed = 0;
    for (auto it = queue_.begin(); it != queue_.end();) {
      if (!predicate(it->item)) {
        ++it;
        continue;
      }
      it = queue_.erase(it);
      ++removed;
      ++stats_.cancelled;
    }
    lock.unlock();
    if (removed > 0) {
      not_full_cv_.notify_all();
    }
    return removed;
  }

  bool waitPop(T* item) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool waited = queue_.empty() && !stopped_;
    const Clock::time_point wait_start = Clock::now();
    if (waited) {
      ++stats_.consumer_wait_count;
      not_empty_cv_.wait(
          lock, [this]() { return stopped_ || !queue_.empty(); });
      stats_.consumer_wait += elapsedNanoseconds(wait_start, Clock::now());
    }
    if (queue_.empty()) {
      return false;
    }
    popLocked(item);
    lock.unlock();
    not_full_cv_.notify_one();
    return true;
  }

  template <typename Rep, typename Period>
  bool waitPopFor(T* item,
                  const std::chrono::duration<Rep, Period>& timeout) {
    Clock::duration bounded_timeout =
        std::chrono::duration_cast<Clock::duration>(timeout);
    if (bounded_timeout < Clock::duration::zero()) {
      bounded_timeout = Clock::duration::zero();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty() && !stopped_) {
      ++stats_.consumer_wait_count;
      const Clock::time_point wait_start = Clock::now();
      const bool ready = not_empty_cv_.wait_for(
          lock,
          bounded_timeout,
          [this]() { return stopped_ || !queue_.empty(); });
      stats_.consumer_wait += elapsedNanoseconds(wait_start, Clock::now());
      if (!ready) {
        ++stats_.consumer_timeout_count;
        return false;
      }
    }
    if (queue_.empty()) {
      return false;
    }
    popLocked(item);
    lock.unlock();
    not_full_cv_.notify_one();
    return true;
  }

  // stop() is terminal. Pending items remain drainable, but all current and
  // future producers are rejected and blocked readers/writers are awakened.
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  bool stopped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
  }

  std::size_t capacity() const { return capacity_; }
  ChannelPolicy policy() const { return policy_; }

  ChannelStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ChannelStats result = stats_;
    result.depth = queue_.size();
    result.stopped = stopped_;
    if (!queue_.empty()) {
      result.oldest_age =
          elapsedNanoseconds(queue_.front().enqueued_at, Clock::now());
    } else {
      result.oldest_age = std::chrono::nanoseconds::zero();
    }
    return result;
  }

 private:
  struct Entry {
    T item;
    Clock::time_point enqueued_at;
    bool replacement_protected = false;
  };

  static std::chrono::nanoseconds elapsedNanoseconds(
      Clock::time_point begin, Clock::time_point end) {
    if (end <= begin) {
      return std::chrono::nanoseconds::zero();
    }
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  }

  bool fullLocked() const {
    return capacity_ > 0 && queue_.size() >= capacity_;
  }

  void enqueueLocked(T item,
                     Clock::time_point now,
                     bool replacement_protected = false) {
    queue_.push_back(
        Entry{std::move(item), now, replacement_protected});
    stats_.high_watermark =
        std::max(stats_.high_watermark, queue_.size());
  }

  PushResult<T> stoppedResult(T item) {
    ++stats_.stopped_pushes;
    PushResult<T> result;
    result.outcome = PushOutcome::kStopped;
    result.unconsumed_item.emplace(std::move(item));
    return result;
  }

  PushResult<T> acceptedResult() {
    ++stats_.accepted;
    PushResult<T> result;
    result.outcome = PushOutcome::kAccepted;
    return result;
  }

  PushResult<T> replacedResult(T replaced,
                               ChannelReplacementReason reason) {
    ++stats_.replaced;
    PushResult<T> result;
    result.outcome = PushOutcome::kReplaced;
    result.replacement_reason = reason;
    result.replaced_item.emplace(std::move(replaced));
    return result;
  }

  PushResult<T> protectedCapacityResult(T item) {
    ++stats_.rejected;
    PushResult<T> result;
    result.outcome = PushOutcome::kRejected;
    result.unconsumed_item.emplace(std::move(item));
    return result;
  }

  typename std::deque<Entry>::iterator oldestReplaceableLocked() {
    return std::find_if(queue_.begin(), queue_.end(), [](const Entry& entry) {
      return !entry.replacement_protected;
    });
  }

  PushResult<T> pushWithPolicy(
      T item,
      ChannelPolicy operation_policy,
      std::optional<Clock::duration> reliable_timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_) {
      return stoppedResult(std::move(item));
    }

    if (operation_policy == ChannelPolicy::kReliableBlocking) {
      return pushReliableLocked(
          std::move(item), reliable_timeout, &lock);
    }

    const Clock::time_point now = Clock::now();
    if (operation_policy == ChannelPolicy::kLatest) {
      if (queue_.empty()) {
        enqueueLocked(std::move(item), now);
        PushResult<T> result = acceptedResult();
        lock.unlock();
        not_empty_cv_.notify_one();
        return result;
      }
      if (queue_.front().replacement_protected) {
        return protectedCapacityResult(std::move(item));
      }
      T replaced = std::move(queue_.front().item);
      queue_.front() = Entry{std::move(item), now, false};
      PushResult<T> result = replacedResult(
          std::move(replaced), ChannelReplacementReason::kLatest);
      lock.unlock();
      not_empty_cv_.notify_one();
      return result;
    }

    if (operation_policy == ChannelPolicy::kLatestByKey) {
      if (!same_key_) {
        throw std::logic_error(
            "LatestByKey operation requires a same-key predicate");
      }
      for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (same_key_(it->item, item)) {
          if (it->replacement_protected) {
            return protectedCapacityResult(std::move(item));
          }
          T replaced = std::move(it->item);
          queue_.erase(it);
          enqueueLocked(std::move(item), now);
          PushResult<T> result = replacedResult(
              std::move(replaced), ChannelReplacementReason::kMatchingKey);
          lock.unlock();
          not_empty_cv_.notify_one();
          return result;
        }
      }
      if (fullLocked()) {
        auto replaceable = oldestReplaceableLocked();
        if (replaceable == queue_.end()) {
          return protectedCapacityResult(std::move(item));
        }
        T replaced = std::move(replaceable->item);
        queue_.erase(replaceable);
        enqueueLocked(std::move(item), now);
        PushResult<T> result = replacedResult(
            std::move(replaced), ChannelReplacementReason::kCapacity);
        lock.unlock();
        not_empty_cv_.notify_one();
        return result;
      }
      enqueueLocked(std::move(item), now);
      PushResult<T> result = acceptedResult();
      lock.unlock();
      not_empty_cv_.notify_one();
      return result;
    }

    if (operation_policy == ChannelPolicy::kRejectNewest && fullLocked()) {
      ++stats_.rejected;
      PushResult<T> result;
      result.outcome = PushOutcome::kRejected;
      result.unconsumed_item.emplace(std::move(item));
      return result;
    }

    if (operation_policy == ChannelPolicy::kDropOldest && fullLocked()) {
      auto replaceable = oldestReplaceableLocked();
      if (replaceable == queue_.end()) {
        return protectedCapacityResult(std::move(item));
      }
      T replaced = std::move(replaceable->item);
      queue_.erase(replaceable);
      enqueueLocked(std::move(item), now);
      PushResult<T> result = replacedResult(
          std::move(replaced), ChannelReplacementReason::kCapacity);
      lock.unlock();
      not_empty_cv_.notify_one();
      return result;
    }

    enqueueLocked(std::move(item), now);
    PushResult<T> result = acceptedResult();
    lock.unlock();
    not_empty_cv_.notify_one();
    return result;
  }

  PushResult<T> pushReliableLocked(
      T item,
      std::optional<Clock::duration> timeout,
      std::unique_lock<std::mutex>* lock) {
    if (fullLocked()) {
      ++stats_.producer_wait_count;
      const Clock::time_point wait_start = Clock::now();
      bool ready = true;
      if (timeout.has_value()) {
        ready = not_full_cv_.wait_for(
            *lock,
            *timeout,
            [this]() { return stopped_ || !fullLocked(); });
      } else {
        not_full_cv_.wait(
            *lock, [this]() { return stopped_ || !fullLocked(); });
      }
      stats_.producer_wait += elapsedNanoseconds(wait_start, Clock::now());
      if (!ready) {
        ++stats_.timed_out;
        PushResult<T> result;
        result.outcome = PushOutcome::kTimedOut;
        result.unconsumed_item.emplace(std::move(item));
        return result;
      }
      if (stopped_) {
        return stoppedResult(std::move(item));
      }
    }

    enqueueLocked(std::move(item), Clock::now(), true);
    PushResult<T> result = acceptedResult();
    lock->unlock();
    not_empty_cv_.notify_one();
    return result;
  }

  void popLocked(T* item) {
    const Clock::time_point now = Clock::now();
    const std::chrono::nanoseconds age =
        elapsedNanoseconds(queue_.front().enqueued_at, now);
    *item = std::move(queue_.front().item);
    queue_.pop_front();
    ++stats_.dequeued;
    stats_.last_dequeue_age = age;
    stats_.max_dequeue_age = std::max(stats_.max_dequeue_age, age);
  }

  const std::size_t capacity_;
  const ChannelPolicy policy_;
  const SameKey same_key_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_cv_;
  std::condition_variable not_full_cv_;
  std::deque<Entry> queue_;
  bool stopped_ = false;
  ChannelStats stats_;
};

// Transitional compatibility alias. Existing code constructs this type with a
// capacity and calls pushDropOldest()/tryPop()/waitPopFor()/stop().
template <typename T>
using ThreadSafeQueue = BoundedChannel<T>;

}  // namespace roomie
