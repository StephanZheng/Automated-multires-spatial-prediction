#include "core/SpatialEntropy.h"

SpatialEntropy::SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float max_value) {
  init(n_spatial_cells, n_bins, min_value, max_value);
}

void SpatialEntropy::init(int n_spatial_cells, int n_bins, float min_value, float max_value) {
  float bin_width = (max_value - min_value) / n_bins;
  init(n_spatial_cells, n_bins, min_value, max_value, bin_width);
}

void SpatialEntropy::init(int n_spatial_cells, int n_bins, float min_value, float max_value, float bin_width) {

  // PrintFancy() << "Hi! Initializing spatial entropy now." << endl;
  // PrintFancy() << "n_spatial_cells " << n_spatial_cells << endl;
  // PrintFancy() << "n_bins " << n_bins << endl;
  // PrintFancy() << "min_value " << min_value << endl;
  // PrintFancy() << "max_value " << max_value << endl;
  // PrintFancy() << "bin_width " << bin_width << endl;

  name = "SpatialEntropy_" + to_string(n_spatial_cells) + "cells_" + to_string(n_bins) + "bins";

  // Every spatial cell has a histogram associated to it.
  histograms_.resize(n_spatial_cells);

  // Initialize all histograms.
  // Note: for(auto histogram : histograms) creates an ephemeral copy, not a reference!
  int kk = 0;
  for (int i = 0; i < histograms_.size(); i++) {
    histograms_[i].histogram_.name = "Histogram_float_cell" + to_string(kk);
    histograms_[i].init(n_bins, min_value, max_value, bin_width);
    histograms_[i].DisableDebugMode();
    kk++;
  }

  // Initialize entropy.
  cell_entropy_.init(n_spatial_cells);
  cell_entropy_.name = "cell_entropy";
}

void SpatialEntropy::AddGradientToHistogram(int cell_index, float gradient) {
  // We have n_spatial_cells
  if (debug_mode) PrintFancy() << name << "::AddGradientToSpatialEntropy cell " << cell_index << " grad " << gradient << endl;
  TEST_LT(cell_index, (int)histograms_.size());
  if (cell_index >= 0 and cell_index < histograms_.size()){
    histograms_[cell_index].Add(gradient);
  }
}

void SpatialEntropy::ComputeEmpiricalDistribution() {
  for (int cell_index = 0; cell_index < histograms_.size(); cell_index++) {
    histograms_[cell_index].ComputeProbabilities();
  }
}
void SpatialEntropy::ComputeSpatialEntropy() {
  for (int cell_index = 0; cell_index < histograms_.size(); cell_index++) {
    histograms_[cell_index].ComputeEntropy();
  }
}

float SpatialEntropy::GetAverageSpatialEntropy() {
  float average_entropy = 0.;
  for (int cell_index = 0; cell_index < histograms_.size(); cell_index++) {
    average_entropy = (average_entropy * cell_index + histograms_[cell_index].entropy_.last()) / (cell_index + 1.);
  }
  return average_entropy;
}