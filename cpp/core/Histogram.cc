#include "core/Histogram.h"

int Histogram::GetIndex(float value) {
  assert(bin_width_ > 0.);
  // The bin layout is as follows:
  // [ catch-too-small, normal bins, catch-too-large ]
  if (value < min_value_) {
    return 0;
  } // If value falls in the range [min_value, max_value], we assign it to index + 1.
  else if (value >= min_value_ and value < max_value_) {
    return (int)((value - min_value_) / bin_width_) + 1;
  }
  else if (value >= max_value_) {
    assert(n_bins_+2 == histogram_.data.size());
    return n_bins_+2;
  }
  else {
    n_illegal_values_seen_++;
    return -1;
  }
}

void Histogram::Add(float value) {
  int index = GetIndex(value);
  *histogram_.att(index) += 1.;
}