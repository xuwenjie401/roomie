#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "roomie/pipeline/thread_safe_queue.hpp"

namespace roomie {
namespace {

using namespace std::chrono_literals;

TEST(BoundedChannel, ZeroCapacityIsUnbounded) {
  BoundedChannel<int> channel(0, ChannelPolicy::kRejectNewest);

  EXPECT_EQ(channel.push(1).outcome, PushOutcome::kAccepted);
  EXPECT_EQ(channel.push(2).outcome, PushOutcome::kAccepted);
  EXPECT_EQ(channel.push(3).outcome, PushOutcome::kAccepted);
  EXPECT_EQ(channel.size(), 3U);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 1);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 2);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 3);
  EXPECT_FALSE(channel.tryPop(&value));

  const ChannelStats stats = channel.stats();
  EXPECT_EQ(stats.capacity, 0U);
  EXPECT_EQ(stats.high_watermark, 3U);
  EXPECT_EQ(stats.accepted, 3U);
  EXPECT_EQ(stats.dequeued, 3U);
}

TEST(BoundedChannel, CapacityOneDropOldestReturnsDisplacedItem) {
  BoundedChannel<int> channel(1, ChannelPolicy::kDropOldest);

  EXPECT_EQ(channel.push(7).outcome, PushOutcome::kAccepted);
  PushResult<int> replacement = channel.push(8);
  EXPECT_EQ(replacement.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(replacement.replacement_reason,
            ChannelReplacementReason::kCapacity);
  ASSERT_TRUE(replacement.replaced_item.has_value());
  EXPECT_EQ(*replacement.replaced_item, 7);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 8);

  const ChannelStats stats = channel.stats();
  EXPECT_EQ(stats.accepted, 1U);
  EXPECT_EQ(stats.replaced, 1U);
  EXPECT_EQ(stats.high_watermark, 1U);
}

TEST(BoundedChannel, RejectNewestPreservesQueuedItemsAndReturnsInput) {
  BoundedChannel<int> channel(2, ChannelPolicy::kRejectNewest);

  EXPECT_TRUE(channel.push(1).accepted());
  EXPECT_TRUE(channel.push(2).accepted());
  PushResult<int> rejected = channel.push(3);
  EXPECT_EQ(rejected.outcome, PushOutcome::kRejected);
  ASSERT_TRUE(rejected.unconsumed_item.has_value());
  EXPECT_EQ(*rejected.unconsumed_item, 3);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 1);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 2);

  const ChannelStats stats = channel.stats();
  EXPECT_EQ(stats.accepted, 2U);
  EXPECT_EQ(stats.rejected, 1U);
}

TEST(BoundedChannel, ReliableProducerWaitsUntilConsumerMakesSpace) {
  BoundedChannel<int> channel(1, ChannelPolicy::kReliableBlocking);
  ASSERT_TRUE(channel.push(10).accepted());

  std::promise<void> producer_entered;
  std::future<void> producer_entered_future = producer_entered.get_future();
  std::promise<PushResult<int>> completion;
  std::future<PushResult<int>> completion_future = completion.get_future();
  std::thread producer([&]() {
    producer_entered.set_value();
    completion.set_value(channel.pushFor(11, 2s));
  });

  producer_entered_future.wait();
  for (int attempt = 0;
       attempt < 200 && channel.stats().producer_wait_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_EQ(completion_future.wait_for(0ms), std::future_status::timeout);

  int value = 0;
  EXPECT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 10);

  EXPECT_EQ(completion_future.wait_for(2s), std::future_status::ready);
  PushResult<int> result = completion_future.get();
  producer.join();
  EXPECT_EQ(result.outcome, PushOutcome::kAccepted);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 11);

  const ChannelStats stats = channel.stats();
  EXPECT_EQ(stats.producer_wait_count, 1U);
  EXPECT_GT(stats.producer_wait.count(), 0);
}

TEST(BoundedChannel, ReliablePushForTimesOutAndReturnsInput) {
  BoundedChannel<int> channel(1, ChannelPolicy::kReliableBlocking);
  ASSERT_TRUE(channel.push(1).accepted());

  PushResult<int> result = channel.pushFor(2, 2ms);
  EXPECT_EQ(result.outcome, PushOutcome::kTimedOut);
  ASSERT_TRUE(result.unconsumed_item.has_value());
  EXPECT_EQ(*result.unconsumed_item, 2);
  EXPECT_EQ(channel.size(), 1U);

  const ChannelStats stats = channel.stats();
  EXPECT_EQ(stats.timed_out, 1U);
  EXPECT_EQ(stats.producer_wait_count, 1U);
}

TEST(BoundedChannel, StopWakesBlockedProducerAndReturnsInput) {
  BoundedChannel<int> channel(1, ChannelPolicy::kReliableBlocking);
  ASSERT_TRUE(channel.push(1).accepted());

  std::promise<void> producer_entered;
  std::future<void> producer_entered_future = producer_entered.get_future();
  std::promise<PushResult<int>> completion;
  std::future<PushResult<int>> completion_future = completion.get_future();
  std::thread producer([&]() {
    producer_entered.set_value();
    completion.set_value(channel.pushFor(2, 5s));
  });

  producer_entered_future.wait();
  for (int attempt = 0;
       attempt < 200 && channel.stats().producer_wait_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  channel.stop();

  EXPECT_EQ(completion_future.wait_for(2s), std::future_status::ready);
  PushResult<int> result = completion_future.get();
  producer.join();
  EXPECT_EQ(result.outcome, PushOutcome::kStopped);
  ASSERT_TRUE(result.unconsumed_item.has_value());
  EXPECT_EQ(*result.unconsumed_item, 2);

  int value = 0;
  EXPECT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 1);
  EXPECT_FALSE(channel.waitPopFor(&value, 1s));
}

TEST(BoundedChannel, StopWakesBlockedConsumer) {
  BoundedChannel<int> channel(1, ChannelPolicy::kReliableBlocking);

  std::promise<void> consumer_entered;
  std::future<void> consumer_entered_future = consumer_entered.get_future();
  std::promise<bool> completion;
  std::future<bool> completion_future = completion.get_future();
  std::thread consumer([&]() {
    int value = 0;
    consumer_entered.set_value();
    completion.set_value(channel.waitPopFor(&value, 5s));
  });

  consumer_entered_future.wait();
  for (int attempt = 0;
       attempt < 200 && channel.stats().consumer_wait_count == 0;
       ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  channel.stop();

  EXPECT_EQ(completion_future.wait_for(2s), std::future_status::ready);
  EXPECT_FALSE(completion_future.get());
  consumer.join();
}

TEST(BoundedChannel, LatestAlwaysKeepsOneNewestItem) {
  BoundedChannel<int> channel(8, ChannelPolicy::kLatest);

  EXPECT_EQ(channel.push(1).outcome, PushOutcome::kAccepted);
  PushResult<int> result = channel.push(2);
  EXPECT_EQ(result.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(result.replacement_reason, ChannelReplacementReason::kLatest);
  ASSERT_TRUE(result.replaced_item.has_value());
  EXPECT_EQ(*result.replaced_item, 1);
  EXPECT_EQ(channel.size(), 1U);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 2);
}

struct KeyedValue {
  std::string key;
  int value = 0;
};

TEST(BoundedChannel, LatestByKeyCoalescesAndPreservesOtherKeys) {
  BoundedChannel<KeyedValue> channel(
      2,
      ChannelPolicy::kLatestByKey,
      [](const KeyedValue& lhs, const KeyedValue& rhs) {
        return lhs.key == rhs.key;
      });

  EXPECT_TRUE(channel.push(KeyedValue{"a", 1}).accepted());
  EXPECT_TRUE(channel.push(KeyedValue{"b", 1}).accepted());

  PushResult<KeyedValue> same_key = channel.push(KeyedValue{"a", 2});
  EXPECT_EQ(same_key.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(same_key.replacement_reason,
            ChannelReplacementReason::kMatchingKey);
  ASSERT_TRUE(same_key.replaced_item.has_value());
  EXPECT_EQ(same_key.replaced_item->key, "a");
  EXPECT_EQ(same_key.replaced_item->value, 1);

  // Replacing a key moves its newest value to the back, so another key is not
  // starved merely because the first key updates frequently.
  KeyedValue value;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value.key, "b");
  EXPECT_EQ(value.value, 1);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value.key, "a");
  EXPECT_EQ(value.value, 2);
}

TEST(BoundedChannel, LatestByKeyDropsOldestKeyWhenCapacityIsFull) {
  BoundedChannel<KeyedValue> channel(
      2,
      ChannelPolicy::kLatestByKey,
      [](const KeyedValue& lhs, const KeyedValue& rhs) {
        return lhs.key == rhs.key;
      });

  EXPECT_TRUE(channel.push(KeyedValue{"a", 1}).accepted());
  EXPECT_TRUE(channel.push(KeyedValue{"b", 1}).accepted());
  PushResult<KeyedValue> result = channel.push(KeyedValue{"c", 1});
  EXPECT_EQ(result.outcome, PushOutcome::kReplaced);
  EXPECT_EQ(result.replacement_reason,
            ChannelReplacementReason::kCapacity);
  ASSERT_TRUE(result.replaced_item.has_value());
  EXPECT_EQ(result.replaced_item->key, "a");

  KeyedValue value;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value.key, "b");
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value.key, "c");
}

TEST(BoundedChannel, ReportsSteadyClockQueueAge) {
  BoundedChannel<int> channel(2);
  ASSERT_TRUE(channel.push(42).accepted());
  std::this_thread::sleep_for(2ms);

  EXPECT_GT(channel.stats().oldest_age.count(), 0);
  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  const ChannelStats stats = channel.stats();
  EXPECT_GT(stats.last_dequeue_age.count(), 0);
  EXPECT_EQ(stats.max_dequeue_age, stats.last_dequeue_age);
}

TEST(BoundedChannel, LegacyThreadSafeQueueSurfaceRemainsAvailable) {
  ThreadSafeQueue<int> queue(1);
  EXPECT_TRUE(queue.pushDropOldest(1));
  EXPECT_TRUE(queue.pushDropOldest(2));

  int value = 0;
  ASSERT_TRUE(queue.waitPopFor(&value, 1ms));
  EXPECT_EQ(value, 2);
  queue.stop();
  EXPECT_FALSE(queue.pushDropOldest(3));
}

TEST(BoundedChannel, ReliableItemCannotBeEvictedByLaterLossyPushes) {
  BoundedChannel<int> channel(2, ChannelPolicy::kDropOldest);
  ASSERT_TRUE(channel.pushUsingPolicy(10, ChannelPolicy::kReliableBlocking));
  ASSERT_TRUE(channel.push(20));

  const PushResult<int> replaced = channel.push(30);
  ASSERT_EQ(replaced.outcome, PushOutcome::kReplaced);
  ASSERT_TRUE(replaced.replaced_item);
  EXPECT_EQ(*replaced.replaced_item, 20);

  const PushResult<int> rejected = channel.push(40);
  ASSERT_EQ(rejected.outcome, PushOutcome::kReplaced);
  ASSERT_TRUE(rejected.replaced_item);
  EXPECT_EQ(*rejected.replaced_item, 30);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 10);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 40);
}

TEST(BoundedChannel, LossyPushRejectsWhenCapacityContainsOnlyReliableItems) {
  BoundedChannel<int> channel(1, ChannelPolicy::kDropOldest);
  ASSERT_TRUE(channel.pushUsingPolicy(10, ChannelPolicy::kReliableBlocking));
  const PushResult<int> rejected = channel.push(20);
  EXPECT_EQ(rejected.outcome, PushOutcome::kRejected);
  ASSERT_TRUE(rejected.unconsumed_item);
  EXPECT_EQ(*rejected.unconsumed_item, 20);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 10);
}

TEST(BoundedChannel, LifecycleCancellationRemovesProtectedBarrier) {
  BoundedChannel<KeyedValue> channel(
      3,
      ChannelPolicy::kDropOldest,
      [](const KeyedValue& lhs, const KeyedValue& rhs) {
        return lhs.key == rhs.key;
      });
  ASSERT_TRUE(channel.pushUsingPolicy(
      KeyedValue{"stale-candidate", 1},
      ChannelPolicy::kReliableBlocking));
  ASSERT_TRUE(channel.push(KeyedValue{"ordinary", 2}));

  EXPECT_EQ(channel.cancelIf([](const KeyedValue& value) {
              return value.key == "stale-candidate";
            }),
            1U);
  EXPECT_EQ(channel.stats().cancelled, 1U);
  EXPECT_EQ(channel.size(), 1U);

  KeyedValue remaining;
  ASSERT_TRUE(channel.tryPop(&remaining));
  EXPECT_EQ(remaining.key, "ordinary");
}

TEST(BoundedChannel, BarrierAdmissionSamplesOldWorkWithoutReorderingRetainedWork) {
  BoundedChannel<int> channel(8, ChannelPolicy::kDropOldest);
  ASSERT_TRUE(channel.push(1));
  ASSERT_TRUE(channel.push(2));
  ASSERT_TRUE(channel.push(3));
  ASSERT_TRUE(channel.push(4));

  ASSERT_TRUE(channel.pushBarrierFor(99, /*max_replaceable_ahead=*/1U, 10ms));
  const ChannelStats admitted = channel.stats();
  EXPECT_EQ(admitted.barrier_shed, 3U);
  EXPECT_EQ(admitted.depth, 2U);

  int value = 0;
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 4);
  ASSERT_TRUE(channel.tryPop(&value));
  EXPECT_EQ(value, 99);
}

}  // namespace
}  // namespace roomie
