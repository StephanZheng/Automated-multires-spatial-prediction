#include "test/Test.h"

using namespace std;

template <typename T>
bool TEST_LT(T a, T b) {
  if (a < b) {
    return true;
  } else {
    return false;
  }
}