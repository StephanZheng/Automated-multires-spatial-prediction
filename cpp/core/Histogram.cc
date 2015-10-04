#include "core/Histogram.h"

int Histogram::GetIndex(float value) {
  assert(bin_width_ > 0.);
  // The bin layout is as follows:
  // [ catch-too-small, normal bins, catch-too-large ]
  if (value < min_value_) {
    return 0;
  } // If value falls in the range [min_value, max_value], we assign it to index + 1.
  else if (value >= min_value_ and value < max_value_) {
    int index = (int)((value - min_value_) / bin_width_) + 1;
    if (index >= n_bins_-1) {
      return n_bins_-2;
    } // Index should not fall in the last catch-all bin,
      // based on the numerical value. So in that case, return the last legal bin.
    return index;
  }
  else if (value >= max_value_) {
    if (n_bins_ != histogram_.size()) {
      showProperties();
    }
    assert(n_bins_ == histogram_.size());
    return n_bins_-1;
  }
  else {
    n_illegal_values_seen_++;
    PrintFancy() << "Illegal value: " << value << " | n_illegal_values_seen: " << n_illegal_values_seen_ << endl;
    showProperties();
    return -1;
  }
}

void Histogram::Add(float value) {
  int index = GetIndex(value);
  if (index >= 0 and index < histogram_.size()) {
    *histogram_.att(index) += 1.;
  }

  // Diagnostics
  if (debug_mode) {
    if (value < min_value_seen_) min_value_seen_ = value;
    if (value > max_value_seen_) max_value_seen_ = value;
    if (value < min_legal_value_seen_ and value > min_value_) min_legal_value_seen_ = value;
    if (value > max_legal_value_seen_ and value < max_value_) max_legal_value_seen_ = value;
  }
}

void Histogram::ComputeProbabilities() {
  float total = GetLegalRangeTotal();
  // Only loop over the bins of the legal range: not 0 and the last one.
  for (int i = 0; i < n_bins_ - 2; i++) {
    *probabilities_.att(i) = histogram_.at(i+1) / total;
  }
}

void Histogram::ComputeEntropy() {
  float entropy = 0.;
  // Only loop over the bins of the legal range: not 0 and the last one.
  for (int i = 0; i < probabilities_.size(); i++) {
    float probability = probabilities_.at(i);
    CheckIsProbability(probability);

    // Only compute entropy if log(prob) is not going to cause overflow.
    if (probability > log_stability_threshold) {
      entropy += probability * std::log(probability);
    }
  }

  PrintFancy() << histogram_.name << "::Entropy = " << entropy << endl;
  entropy_.push_back(entropy);
  entropy_.showContents(20, 10);
}

void Histogram::CheckIsProbability(float probability) {
  TEST_LE(probability, static_cast<float>(1.0));
  TEST_GE(probability, static_cast<float>(0.0));
}