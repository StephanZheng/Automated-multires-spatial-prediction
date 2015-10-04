#ifndef HISTOGRAM_H
#define HISTOGRAM_H

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

#include "core/DataBlob.h"
#include "util/PrettyOutput.h"

class Histogram {
public:
  VectorBlob histogram_;

  // Histogram also has a counter "n_illegal_values_seen" for values that are not of the right type (NaNs etc).
  int n_illegal_values_seen_;

  int n_bins_;
  float bin_width_;
  float min_value_;
  float max_value_;

  Histogram() {};
  void init(int n_bins, float min_value, float bin_width) {

    // Histogram adds 2 bins for values that lie outside the range.
    histogram_.init(n_bins+2);
    n_bins_ = n_bins;
    min_value_ = min_value;
    bin_width_ = bin_width;
    max_value_ = min_value + n_bins * bin_width;
    showProperties();
  }
  int GetIndex(float value);
  void Add(float value);
  void showProperties() {
    PrintFancy() << "Histogram [" << histogram_.name << "]" << endl;
    PrintFancy() << "n_bins " << n_bins_ << endl;
    PrintFancy() << "min_value " << min_value_ << endl;
    PrintFancy() << "bin_width " << bin_width_ << endl;
  }
  void showContents(int lim, int lb) {
    showProperties();
    histogram_.showVectorContents(lim, lb);
  }
};

#endif