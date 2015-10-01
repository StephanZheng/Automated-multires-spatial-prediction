// Copyright 2014 Stephan Zheng

#ifndef QUEUEMESSAGE_H
#define QUEUEMESSAGE_H

#include <vector>
#include <string>
#include <queue>
#include <chrono>
#include <boost/timer.hpp>

using std::chrono::high_resolution_clock;

#include "config/GlobalConstants.h"
#include "util/PrettyOutput.h"

class QueueMessageComplete {
  public:
  unsigned int  task_type;
};

class QueueMessage {
  public:
  unsigned int  task_type;

  //  For computing weight / bias updates
  int       index_A;
  int       frame_id;
  int       ground_truth_label;

  //  For computing loss
  int       labels_id_start;
  int       labels_id_end;
  int       index_A_start;
  int       index_A_end;
};

class TaskQueue {
public:
  int name;
  std::queue<QueueMessage>  taskQueue;
  std::queue<int>           taskCompletionQueue;
  boost::mutex              mutex_;
  boost::mutex              mutex_complete_;
  boost::condition_variable qComplete, qGoFetch;

  TaskQueue() {}

  void getTaskCompleteMessage() {
    boost::mutex::scoped_lock lock(mutex_complete_);
    while (taskCompletionQueue.size() == 0) qComplete.wait(lock);
    taskCompletionQueue.pop();
  }
  void getTaskMessage(int thread_id, QueueMessage *qm, int MiniBatchSize) {
    boost::mutex::scoped_lock lock(mutex_);
    while (taskQueue.size() == 0) qGoFetch.wait(lock);
    for (int i = 0; i < MiniBatchSize; ++i) {
      qm[i] = taskQueue.front();
      taskQueue.pop();
    }
  }
  void waitForTasksToComplete(int threshold) {
    int n_completed_tasks_seen = 0;
    while (1) {
      getTaskCompleteMessage();
      n_completed_tasks_seen++;
      if (n_completed_tasks_seen >= threshold) {
        // PrintFancy(high_resolution_clock::now(), "All workers finished for this task!");
        break;
      }
    }
  }
};

#endif