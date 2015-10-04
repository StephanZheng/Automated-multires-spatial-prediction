// Copyright 2014 Stephan Zheng

#ifndef TEST_H
#define TEST_H

#include <chrono>
#include <time.h>
#include <random>

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <queue>

#include "util/PrettyOutput.h"

using namespace std;

bool TEST_LT(int a, int b);
bool TEST_LT(float a, float b);
bool TEST_LT(double a, double b);
template <typename T>
bool TEST_LT_(T a, T b);

bool TEST_GT(int a, int b);
bool TEST_GT(float a, float b);
bool TEST_GT(double a, double b);
template <typename T>
bool TEST_GT_(T a, T b);

bool TEST_LE(int a, int b);
bool TEST_LE(float a, float b);
bool TEST_LE(double a, double b);
template <typename T>
bool TEST_LE_(T a, T b);

bool TEST_GE(int a, int b);
bool TEST_GE(float a, float b);
bool TEST_GE(double a, double b);
template <typename T>
bool TEST_GE_(T a, T b);

bool TEST_EQ(int a, int b);
template <typename T>
bool TEST_EQ_(T a, T b);

#endif