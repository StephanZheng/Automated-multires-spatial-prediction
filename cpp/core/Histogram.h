#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "core/DataBlob.h"
#include "util/IOController.h"
#include "util/PrettyOutput.h"
#include "test/Test.h"

class Histogram {
public:

  bool debug_mode;
  void EnableDebugMode() {debug_mode = true;};
  void DisableDebugMode() {debug_mode = false;};

  // VectorBlob index ranges over spatial cells.
  VectorBlob histogram_;
  VectorBlob probabilities_;

  // Index loops over time. Every historical entropy value is stored here.
  VectorBlob entropy_;

  // Histogram also has a counter "n_illegal_values_seen" for values that are not of the right type (NaNs etc).
  int n_illegal_values_seen_;
  int n_outliers_lower_;
  int n_outliers_upper_;

  float log_stability_threshold;

  int n_bins_;
  float bin_width_;
  float min_value_;
  float max_value_;

  float min_value_seen_;
  float max_value_seen_;
  float min_legal_value_seen_;
  float max_legal_value_seen_;

  Histogram() {};
  void init(int n_bins, float min_value, float bin_width) {
    // To compute the legal range, do not include the two catch-all bins.
    int max_value = min_value + n_bins * bin_width;
    max_value_ = max_value;
    init(n_bins, min_value, max_value, bin_width);
  }
  void init(int n_bins, float min_value, float max_value, float bin_width) {

    // Sanity-check the input.
    assert(n_bins > 0); // We need at least 1 bin.
    assert(max_value > min_value);

    // Histogram adds 2 bins for values that lie outside the range.
    histogram_.init(n_bins);
    probabilities_.init(n_bins);

    min_value_ = min_value;
    max_value_ = max_value;
    bin_width_ = bin_width;

    // Note that we include the catch all bins here.
    // The convention is that n_bins == histogram_.size().
    n_bins_ = n_bins;

    // To prevent log overflow.
    log_stability_threshold = 1e-20;

    // Initialize entropy holder to empty!
    entropy_.init(0);

    // showProperties();
  }

  // Histogram of counts.
  int GetIndex(float value);
  void Add(float value);

  // Statistics
  void RecordExtrema(float value);
  void ComputeProbabilities();
  void ComputeEntropy();

  void EraseProbabilities() {
    probabilities_.erase();
  }
  void EraseHistogram() {
    histogram_.erase();
  }

  // Utility functions
  int size() const {
    return histogram_.size();
  }
  int GetNumberOfLegalBins() {
    assert(n_bins_ > 0); // We need at least 1 + 2 bins (2 catch-all and 1 legal)
    return n_bins_;
  }
  int GetNumberOfAllBins() {
    TEST_EQ(n_bins_, size());
    return n_bins_;
  }
  float GetLegalRangeTotal() {
    float count = 0;
    TEST_GE(n_bins_, 1);
    for (int i = 0; i < n_bins_; i++) {
      count += histogram_.at(i);
    }
    return count;
  }
  void WriteProbabilityToFile(string fn_probability) {
    for (auto prob : probabilities_.data) {
      WriteToFile(fn_probability, to_string(prob) + ",");
    }
    WriteToFile(fn_probability, to_string('\n'));
  }

  // Diagnostics
  bool CheckIsProbability(float probability);
  float ShowEntropy();
  void showProperties() {
    PrintFancy() << "Properties of histogram [" << histogram_.name << "]" << endl;
    PrintFancy() << "n_bins (includes catch-all) " << n_bins_ << endl;
    PrintFancy() << "min_value " << min_value_ << endl;
    PrintFancy() << "max_value " << max_value_ << endl;
    PrintFancy() << "bin_width " << bin_width_ << endl;
    PrintFancy() << "log_stability_threshold " << log_stability_threshold << endl;
  }
  void showContents(int lim, int lb) {
    showProperties();
    histogram_.showVectorContents(lim, lb);
  }
};

#endif