#include "core/SpatialEntropy.h"

SpatialEntropy::SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float bin_width) {
  init(n_spatial_cells, n_bins, min_value, bin_width);
}

void SpatialEntropy::init(int n_spatial_cells, int n_bins, float min_value, float bin_width) {

  PrintFancy() << "Hi! Initializing spatial entropy now." << endl;
  PrintFancy() << "n_spatial_cells " << n_spatial_cells << endl;
  PrintFancy() << "n_bins " << n_bins << endl;
  PrintFancy() << "min_value " << min_value << endl;
  PrintFancy() << "bin_width " << bin_width << endl;

  // Every spatial cell has a histogram associated to it.
  histograms_float_.resize(n_spatial_cells);
  histograms_sign_.resize(n_spatial_cells);

  // Initialize all histograms.
  // Note: for(auto histogram : histograms) creates an ephemeral copy, not a reference!
  int kk = 0;
  for (int i = 0; i < histograms_float_.size(); i++) {
    histograms_float_[i].histogram_.name = "Histogram_float_cell" + to_string(kk);
    histograms_float_[i].init(n_bins, min_value, bin_width);
    kk++;
  }
  kk = 0;
  for (int i = 0; i < histograms_sign_.size(); i++) {
    histograms_sign_[i].histogram_.name = "Histogram_sign_cell" + to_string(kk);
    histograms_sign_[i].init(2, -1., 1.);
    kk++;
  }

  // Initialize entropy.
  cell_entropy_.init(n_spatial_cells);
  cell_entropy_.name = "cell_entropy";
}


void SpatialEntropy::AddGradientToSpatialEntropy(int cell_index, float gradient) {
  cout << cell_index << " " << gradient;
  assert(cell_index < histograms_float_.size());
  histograms_float_[cell_index].Add(gradient);
}
void SpatialEntropy::AddGradientSignToSpatialEntropy(int cell_index, float gradient) {
  cout << cell_index << " " << gradient;
  assert(cell_index < histograms_sign_.size());
  histograms_float_[cell_index].Add(gradient);
}

float SpatialEntropy::GetAverageSpatialEntropy() {
  return 0.;
}