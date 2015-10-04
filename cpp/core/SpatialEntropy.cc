#include "core/SpatialEntropy.h"

SpatialEntropy::SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float max_value) {
  init(n_spatial_cells, n_bins, min_value, max_value);
}

void SpatialEntropy::init(int n_spatial_cells, int n_bins, float min_value, float max_value) {
  float bin_width = (max_value - min_value) / n_bins;
  init(n_spatial_cells, n_bins, min_value, max_value, bin_width);
}

void SpatialEntropy::init(int n_spatial_cells, int n_bins, float min_value, float max_value, float bin_width) {

  PrintFancy() << "Hi! Initializing spatial entropy now." << endl;
  PrintFancy() << "n_spatial_cells " << n_spatial_cells << endl;
  PrintFancy() << "n_bins " << n_bins << endl;
  PrintFancy() << "min_value " << min_value << endl;
  PrintFancy() << "max_value " << max_value << endl;
  PrintFancy() << "bin_width " << bin_width << endl;

  name = "SpatialEntropy_" + to_string(n_spatial_cells) + "bins";

  // Every spatial cell has a histogram associated to it.
  histograms_float_.resize(n_spatial_cells);
  histograms_sign_.resize(n_spatial_cells);

  // Initialize all histograms.
  // Note: for(auto histogram : histograms) creates an ephemeral copy, not a reference!
  int kk = 0;
  for (int i = 0; i < histograms_float_.size(); i++) {
    histograms_float_[i].histogram_.name = "Histogram_float_cell" + to_string(kk);
    histograms_float_[i].init(n_bins, min_value, max_value, bin_width);
    histograms_float_[i].DisableDebugMode();
    kk++;
  }
  kk = 0;
  for (int i = 0; i < histograms_sign_.size(); i++) {
    histograms_sign_[i].histogram_.name = "Histogram_sign_cell" + to_string(kk);
    histograms_sign_[i].init(2, -1., 1., 1.);
    histograms_sign_[i].DisableDebugMode();
    kk++;
  }

  // Initialize entropy.
  cell_entropy_.init(n_spatial_cells);
  cell_entropy_.name = "cell_entropy";
}


void SpatialEntropy::AddGradientToHistogram(int cell_index, float gradient) {
  // We have n_spatial_cells
  TEST_LT(cell_index, (int)histograms_float_.size());
  if (debug_mode) PrintFancy() << name << "::AddGradientToSpatialEntropy cell " << cell_index << " grad " << gradient << endl;
  if (cell_index >= 0 and cell_index < histograms_float_.size()){
    histograms_float_[cell_index].Add(gradient);
  }
}
void SpatialEntropy::AddGradientSignToHistogram(int cell_index, float gradient) {
  TEST_LT(cell_index, (int)histograms_sign_.size());
  if (debug_mode) PrintFancy() << name << "::AddGradientSignToSpatialEntropy cell " << cell_index << " grad " << gradient << endl;
  if (cell_index >= 0 and cell_index < histograms_sign_.size()){
    histograms_sign_[cell_index].Add(gradient);
  }
}

void SpatialEntropy::ComputeEmpiricalDistribution(std::vector<Histogram> histograms) {
  for (int cell_index = 0; cell_index < histograms.size(); cell_index++) {
    histograms[cell_index].ComputeProbabilities();
  }
}

void SpatialEntropy::ComputeSpatialEntropy(std::vector<Histogram> histograms) {
  for (int cell_index = 0; cell_index < histograms.size(); cell_index++) {
    histograms[cell_index].ComputeEntropy();
  }
}

float SpatialEntropy::GetAverageSpatialEntropy(const std::vector<Histogram>& histograms) {
  float average_entropy = 0.;
  for (int cell_index = 0; cell_index < histograms.size(); cell_index++) {
    average_entropy = (average_entropy * cell_index + histograms[cell_index].entropy_.last()) / (cell_index + 1.);
  }
  return average_entropy;
}