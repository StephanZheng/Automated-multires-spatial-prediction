#include "core/Histogram.h"

int Histogram::GetIndex(float value) {
  assert(bin_width_ > 0.);
  // The bin layout is as follows:
  // [ catch-too-small, normal bins, catch-too-large ]
  if (value < min_value_) {
    return -1;
  } // If value falls in the range [min_value, max_value], we assign it to index + 1.
  else if (value >= min_value_ and value < max_value_) {
    int index = (int)((value - min_value_) / bin_width_);
    if (index >= n_bins_) {
      return n_bins_-1;
    } // Index should not fall in the last catch-all bin,
      // based on the numerical value. So in that case, return the last legal bin.
    return index;
  }
  else if (value >= max_value_) {
    if (n_bins_ != histogram_.size()) {
      showProperties();
    }
    assert(n_bins_ == histogram_.size());
    return n_bins_;
  }
  else {
    n_illegal_values_seen_++;
    PrintFancy() << "Illegal value: " << value << " | n_illegal_values_seen: " << n_illegal_values_seen_ << endl;
    showProperties();
    return -2;
  }
}

void Histogram::Add(float value) {
  int index = GetIndex(value);
  assert(n_bins_ == histogram_.size());
  if (index >= 0 and index < histogram_.size()) {
    *histogram_.att(index) += 1.;
  } else if (index == -1) {
    n_outliers_lower_++;
  } else if (index == n_bins_) {
    n_outliers_upper_++;
  } else if (index == -2) {
    n_illegal_values_seen_++;
  }
}

void Histogram::RecordExtrema(float value) {
  // Diagnostics
  if (value < min_value_seen_) min_value_seen_ = value;
  if (value > max_value_seen_) max_value_seen_ = value;
  if (value < min_legal_value_seen_ and value > min_value_) min_legal_value_seen_ = value;
  if (value > max_legal_value_seen_ and value < max_value_) max_legal_value_seen_ = value;

  PrintFancy() << "min_value_seen_: " << min_value_seen_;
  cout << "max_value_seen_: " << max_value_seen_;
  cout << "min_legal_value_seen_: " << min_legal_value_seen_;
  cout << "max_legal_value_seen_: " << max_legal_value_seen_;
  cout << endl;
}

void Histogram::ComputeProbabilities() {
  float total = GetLegalRangeTotal();
  if (!(total > 0.)) {
    total = 1.;
  }
  // Only loop over the bins of the legal range: not 0 and the last one.
  TEST_GE(n_bins_, 1); // We need at least 1 bin.
  assert(histogram_.size() == n_bins_);
  for (int i = 0; i < n_bins_; i++) {
    *probabilities_.att(i) = histogram_.at(i) / total;
  }
}

void Histogram::ComputeEntropy() {
  float entropy = 0.;
  // Only loop over the bins of the legal range: not 0 and the last one.
  for (int i = 0; i < probabilities_.size(); i++) {
    float probability = probabilities_.at(i);
    if (!CheckIsProbability(probability)) {
      probability = log_stability_threshold;
    };

    // Only compute entropy if log(prob) is not going to cause overflow.
    if (probability > log_stability_threshold) {
      entropy += probability * std::log(probability);
    }
  }
  entropy_.push_back(entropy);
  if (debug_mode) {
    PrintFancy() << histogram_.name << "::Entropy = " << entropy << endl;
    entropy_.showContents(10, 1);
  }
}

float Histogram::ShowEntropy() {
  PrintFancy() << histogram_.name << "::Entropy = " << entropy_.last() << endl;
  return entropy_.last();
}

bool Histogram::CheckIsProbability(float probability) {
  if (TEST_LE(probability, static_cast<float>(1.0)) \
      and TEST_GE(probability, static_cast<float>(0.0)) ) {
    return true;
  } else {
    return false;
  }
}