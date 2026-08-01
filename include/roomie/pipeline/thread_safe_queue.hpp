#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace roomie {

template <typename T>
class ThreadSafeQueue {
 public:
  explicit ThreadSafeQueue(std::size_t max_size) : max_size_(max_size) {}

  bool pushDropOldest(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return false;
      }
      while (max_size_ > 0 && queue_.size() >= max_size_) {
        queue_.pop_front();
      }
      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
    return true;
  }

  bool tryPop(T* item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    *item = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  template <typename Rep, typename Period>
  bool waitPopFor(T* item, const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, timeout, [this]() { return stopped_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    *item = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    cv_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  bool stopped_ = false;
};

}  // namespace roomie
