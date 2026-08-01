#include "roomie/pipeline/worker_thread.hpp"

#include <utility>

namespace roomie {

WorkerThread::WorkerThread(std::string name) : name_(std::move(name)) {}

WorkerThread::~WorkerThread() { stop(); }

void WorkerThread::start() {
  if (running_) {
    return;
  }
  stop_requested_ = false;
  running_ = true;
  thread_ = std::thread([this]() {
    run();
    running_ = false;
  });
}

void WorkerThread::stop() {
  if (!running_ && !thread_.joinable()) {
    return;
  }
  stop_requested_ = true;
  onStopRequested();
  if (thread_.joinable()) {
    thread_.join();
  }
  running_ = false;
}

bool WorkerThread::running() const { return running_; }

const std::string& WorkerThread::name() const { return name_; }

bool WorkerThread::stopRequested() const { return stop_requested_; }

void WorkerThread::onStopRequested() {}

}  // namespace roomie
