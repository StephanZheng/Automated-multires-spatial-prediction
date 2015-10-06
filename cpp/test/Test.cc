#include "test/Test.h"

using namespace std;

bool TEST_LT(int a, int b) {
  return TEST_LT_<int>(a, b);
}

bool TEST_LT(float a, float b) {
  return TEST_LT_<float>(a, b);
}

bool TEST_LT(double a, double b) {
  return TEST_LT_<double>(a, b);
}

bool TEST_LT(int a, int b, string m) {
  cout << "Called from: " << m << endl;
  TEST_LT_<int>(a, b);
}

template <typename T>
bool TEST_LT_(T a, T b) {
  if (a < b) {
    return true;
  } else {
    PrintFancy() << "Check a < b failed with a " << a << " b " << b << endl;
    assert(a<b);
    return false;
  }
}

bool TEST_GT(int a, int b) {
  return TEST_GT_<int>(a, b);
}

bool TEST_GT(float a, float b) {
  return TEST_GT_<float>(a, b);
}

bool TEST_GT(double a, double b) {
  return TEST_GT_<double>(a, b);
}

template <typename T>
bool TEST_GT_(T a, T b) {
  if (a > b) {
    return true;
  } else {
    PrintFancy() << "Check a > b failed with a " << a << " b " << b << endl;
    assert(a>b);
    return false;
  }
}

bool TEST_LE(int a, int b) {
  return TEST_LE_<int>(a, b);
}

bool TEST_LE(float a, float b) {
  return TEST_LE_<float>(a, b);
}

bool TEST_LE(double a, double b) {
  return TEST_LE_<double>(a, b);
}

template <typename T>
bool TEST_LE_(T a, T b) {
  if (a <= b) {
    return true;
  } else {
    PrintFancy() << "Check a <= b failed with a " << a << " b " << b << endl;
    assert(a<=b);
    return false;
  }
}

bool TEST_GE(int a, int b) {
  return TEST_GE_<int>(a, b);
}

bool TEST_GE(float a, float b) {
  return TEST_GE_<float>(a, b);
}

bool TEST_GE(double a, double b) {
  return TEST_GE_<double>(a, b);
}

template <typename T>
bool TEST_GE_(T a, T b) {
  if (a >= b) {
    return true;
  } else {
    PrintFancy() << "Check a >= b failed with a " << a << " b " << b << endl;
    return false;
  }
}

bool TEST_EQ(int a, int b) {
  return TEST_EQ_<int>(a, b);
}

template <typename T>
bool TEST_EQ_(T a, T b) {
  if (a == b) {
    return true;
  } else {
    PrintFancy() << "Check a == b failed with a " << a << " b " << b << endl;
    assert(a>=b);
    return false;
  }
}