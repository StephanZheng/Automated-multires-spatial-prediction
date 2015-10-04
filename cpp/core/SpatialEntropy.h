#ifndef SPATIALENTROPY_H
#define SPATIALENTROPY_H

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
#include "core/Histogram.h"
#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"
#include "test/Test.h"

class SpatialEntropy {

// TODO(stz): Redundant class! Entropy is stored inside Histogram, for now. Perhaps refactor later.
// index: float
VectorBlob cell_entropy_;

public:

  // TODO(stz): not good practice, use accessor etc methods. Being lazy here.
  string name;
  bool debug_mode;

  // index x bin: float
  std::vector<Histogram> histograms_;

  SpatialEntropy() {};
  SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float max_value);
  void init(int n_spatial_cells, int n_bins, float min_value, float max_value);
  void init(int n_spatial_cells, int n_bins, float min_value, float max_value, float bin_width);

  void EnableDebugMode() {debug_mode = true;};
  void DisableDebugMode() {debug_mode = false;};

  void AddGradientToHistogram(int index, float gradient);
  void AddGradientSignToHistogram(int cell_index, float gradient);

  // Methods loop over all spatial cells: for every cell, computes the empirical distribution and entropy.
  void ComputeEmpiricalDistribution();
  void ComputeSpatialEntropy();
  float GetAverageSpatialEntropy();

  void ShowHistograms() {
    PrintFancy() << "Showing histograms for [" << name << "]";
    for (int i = 0; i < histograms_.size(); i++) {
      histograms_[i].showProperties();
    }
  }
  void ShowEntropies() {
    for (int i = 0; i < histograms_.size(); i++) {
      histograms_[i].ShowEntropy();
    }
  }
  void ShowSummary() {
    for (int i = 0; i < histograms_.size(); i++) {
      PrintFancy() << "Stats for " << histograms_[i].histogram_.name;
      cout << " mean: " << histograms_[i].histogram_.getMean();
      cout << " var: " << std::sqrt(histograms_[i].histogram_.getVariance()) << endl;
    }
  }

  void LogToFile(high_resolution_clock::time_point start_time,
                 string fn_probability,
                 string fn_entropy) {

    float avg_entropy = GetAverageSpatialEntropy();
    PrintFancy() << "Average spatial entropy: " << avg_entropy << endl;

    // Write average-entropy
    WriteToFile(fn_entropy, to_string(GetTimeElapsedSince(start_time)) + ",");
    WriteToFile(fn_entropy, to_string(avg_entropy) + "\n");

    // Write probabilities - for external KL divergence computation
    WriteToFile(fn_probability, to_string(GetTimeElapsedSince(start_time)) + ",");
    WriteToFile(fn_probability, to_string("probabilities") + "\n");

  }

protected:

private:

};


#endif