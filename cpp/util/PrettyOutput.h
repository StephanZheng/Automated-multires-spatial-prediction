// Copyright 2014 Stephan Zheng

#ifndef PRETTYOUTPUT_H
#define PRETTYOUTPUT_H

#include <string>
#include <chrono>
#include "config/GlobalConstants.h"

// ============================================================================
// Output prettifiers
// ============================================================================
using namespace std;
using std::chrono::high_resolution_clock;

void PrintDelimiter(int lines_before, int lines_after, int length, char c);
void PrintFancy(high_resolution_clock::time_point session_start_time, const string &message);
void PrintFancy(high_resolution_clock::time_point session_start_time, const int &message);
void PrintFancy(high_resolution_clock::time_point session_start_time, const float &message);
void PrintFancy(high_resolution_clock::time_point session_start_time, const float &message, const int width, const int precision);
template <typename T>
void PrintFancy(const T& message);
void PrintWithDelimiters(high_resolution_clock::time_point session_start_time, const string& message);
void PrintWithDelimiters(const string& message);
void PrintTimeElapsedSince(high_resolution_clock::time_point start_time, const string& message);

string StringPadding(string original, int charCount);

#endif