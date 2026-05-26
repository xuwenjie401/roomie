#pragma once

#include <atomic>
#include <string>
#include <thread>

namespace roomie {

class WorkerThread {
 public:
  explicit WorkerThread(std::string name);
  virtual ~WorkerThread();

  WorkerThread(const WorkerThread&) = delete;
  WorkerThread& operator=(const WorkerThread&) = delete;

  void start();
  void stop();

  bool running() const;
  const std::string& name() const;

 protected:
  bool stopRequested() const;
  virtual void run() = 0;
  virtual void onStopRequested();

 private:
  std::string name_;
  std::thread thread_;
  std::atomic_bool stop_requested_{false};
  std::atomic_bool running_{false};
};

}  // namespace roomie
