// Copyright 2014 Stephan Zheng

#ifndef PERFORMANCECONTROLLER_H
#define PERFORMANCECONTROLLER_H

#include <boost/timer.hpp>
#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <chrono>
#include <time.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <fstream>

#include "config/GlobalConstants.h"
#include "config/Settings.h"

#include "core/DataBlob.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"

#include "util/CurrentStateBlob.h"
#include "util/PrettyOutput.h"

using namespace boost;
using namespace std::chrono;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

static int GetTimeElapsedSince(high_resolution_clock::time_point start_time){
  high_resolution_clock::time_point now = high_resolution_clock::now();
  int delta_t = (duration_cast<duration<double> >(now - start_time)).count();
  return delta_t;
}

static void PrintTimeElapsedSince(high_resolution_clock::time_point start_time, string message){
  PrintFancy(start_time, message + " :: " + to_string(GetTimeElapsedSince(start_time)) + " seconds");
}

class PerformanceController {
  Settings *Settings_;

  public:
  PerformanceController(Settings *aSettings_) {
    Settings_ = aSettings_;
  }

  void showTimeElapsed(high_resolution_clock::time_point start_time) {
    high_resolution_clock::time_point now = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double> >(now - start_time);
    PrintFancy(Settings_->session_start_time, "Runtime: " + to_string(time_span.count()) + " seconds");
  }

};

#endif