#ifndef MLP_MODEL_UTILITY_THREAD_POOL_H_
#define MLP_MODEL_UTILITY_THREAD_POOL_H_

#include <algorithm>
#include <future>
#include <iterator>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace s21 {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t num_threads) : stop_{false} {
    for (std::size_t i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this]() {
        while (!stop_) {
          Task task;
          {
            std::unique_lock<std::mutex> lock{mtx_};
            cv_.wait(lock, [this]() { return stop_ or !queue_.empty(); });
            if (stop_ and queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock{mtx_};
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) {
      thread.join();
    }
  }

  template <typename F, typename... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto result = task->get_future();
    {
      std::unique_lock<std::mutex> lock{mtx_};
      if (stop_) throw std::runtime_error("enqueue on stopped ThreadPool");
      auto new_task = [task]() { (*task)(); };
      queue_.emplace(std::move(new_task));
    }
    cv_.notify_one();
    return result;
  }

 private:
  using Task = std::function<void()>;

  std::vector<std::thread> threads_;
  std::queue<Task> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_;
};
}  // namespace s21

#endif  // MLP_MODEL_UTILITY_THREAD_POOL_H_
