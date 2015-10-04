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
  std::vector<Histogram> histograms_float_;
  std::vector<Histogram> histograms_sign_;


  SpatialEntropy() {};
  SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float max_value);
  void init(int n_spatial_cells, int n_bins, float min_value, float max_value);
  void init(int n_spatial_cells, int n_bins, float min_value, float max_value, float bin_width);

  void EnableDebugMode() {debug_mode = true;};
  void DisableDebugMode() {debug_mode = false;};

  void AddGradientToHistogram(int index, float gradient);
  void AddGradientSignToHistogram(int cell_index, float gradient);

  // Methods loop over all spatial cells: for every cell, computes the empirical distribution and entropy.
  void ComputeEmpiricalDistribution(std::vector<Histogram> histograms);
  void ComputeSpatialEntropy(std::vector<Histogram> histograms);

  float GetAverageSpatialEntropy(const std::vector<Histogram>& histograms);

  void ShowHistograms() {
    PrintFancy() << "Showing histograms for [" << name << "]";
    for (int i = 0; i < histograms_float_.size(); i++) {
      histograms_float_[i].showProperties();
    }
  };
  void ShowSummary() {
    for (int i = 0; i < histograms_float_.size(); i++) {
      PrintFancy() << "Stats for " << histograms_float_[i].histogram_.name;
      cout << " mean: " << histograms_float_[i].histogram_.getMean();
      cout << " var: " << std::sqrt(histograms_float_[i].histogram_.getVariance()) << endl;
    }
  }

protected:

private:

};


#endif