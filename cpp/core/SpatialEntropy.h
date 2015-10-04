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

class SpatialEntropy {

// index x bin: float
std::vector<Histogram> histograms_float_;
std::vector<Histogram> histograms_sign_;

// index: float
VectorBlob cell_entropy_;

public:
  SpatialEntropy() {};
  SpatialEntropy(int n_spatial_cells, int n_bins, float min_value, float bin_width);
  void init(int n_spatial_cells, int n_bins, float min_value, float bin_width);

  void AddGradientToSpatialEntropy(int index, float gradient);
  void AddGradientSignToSpatialEntropy(int cell_index, float gradient);
  float GetAverageSpatialEntropy();
  void ShowHistograms() {
    for (auto hist : histograms_float_) {
      hist.showProperties();
    }
  };

protected:

private:

};


#endif