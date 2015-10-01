#ifndef TRAINUTIL_H
#define TRAINUTIL_H

#include <time.h>
#include <random>

inline float giveRandom(float min, float max) {
  std::mt19937 e2(time(0));
  std::uniform_real_distribution<> dist_b(min, max);
  return dist_b(e2);
}
inline float getSign(float x, int randomize_at_zero) {
  float grad_at_zero = 0.0;

  if (randomize_at_zero == 1) {
    grad_at_zero = giveRandom(0.0, 1.0);
  }

  if (x > SPARSE_MAT_S_ZERO_BAND) return 1.0;
  else if (x < SPARSE_MAT_S_ZERO_BAND) return -1.0;
  else return grad_at_zero;
}

#endif