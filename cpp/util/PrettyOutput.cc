// Copyright 2014 Stephan Zheng

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

#include "PrettyOutput.h"

// #include "classes.h"

using namespace std;
using std::chrono::high_resolution_clock;

void PrintDelimiter(int lines_before, int lines_after, int length, char c){
  for (int i = 0; i < lines_before; ++i) cout << endl;
  cout << std::setfill(c) << std::setw(length) << "" << std::setfill(' ') << endl;
  for (int i = 0; i < lines_after; ++i) cout << endl;
}

string getHumanTimeInterval(const high_resolution_clock::time_point s, const high_resolution_clock::time_point n) {

  ostringstream out;

  chrono::duration<float> d = n - s;

  const chrono::hours hours   = chrono::duration_cast<chrono::hours>(d);
  d      -= hours;
  const chrono::minutes minutes = chrono::duration_cast<chrono::minutes>(d);
  d      -= minutes;
  const chrono::seconds seconds = chrono::duration_cast<chrono::seconds>(d);

  out << setfill('0') << setw(4) << setprecision(4) << hours.count() << ':';
  out << setfill('0') << setw(2) << setprecision(2) << minutes.count() << ':';
  out << setfill('0') << setw(2) << setprecision(2) << seconds.count();

  return out.str();
}


void PrintFancy(high_resolution_clock::time_point session_start_time, const string &message) {

  high_resolution_clock::time_point _now = high_resolution_clock::now();
  time_t now = chrono::system_clock::to_time_t(_now);
  char str[80];
  strftime(str, sizeof(str), "%d %b %Y %H:%M:%S", localtime(&now));

  cout << "[" << str << " - " << std::setw(4) << std::setprecision(3) << getHumanTimeInterval(session_start_time, _now) << "s" << "] " << std::setprecision(20) << message << endl;
}

void PrintFancy(high_resolution_clock::time_point session_start_time, const int &message) {

  high_resolution_clock::time_point _now = high_resolution_clock::now();
  time_t now = chrono::system_clock::to_time_t(_now);
  char str[80];
  strftime(str, sizeof(str), "%d %b %Y %H:%M:%S", localtime(&now));

  cout << "[" << str << " - " << std::setw(4) << std::setprecision(3) << getHumanTimeInterval(session_start_time, _now) << "s" << "] " << std::setprecision(20) << message << endl;
}

void PrintFancy(high_resolution_clock::time_point session_start_time, const float &message) {

  high_resolution_clock::time_point _now = high_resolution_clock::now();
  time_t now = chrono::system_clock::to_time_t(_now);
  char str[80];
  strftime(str, sizeof(str), "%d %b %Y %H:%M:%S", localtime(&now));

  cout << "[" << str << " - " << std::setw(4) << std::setprecision(3) << getHumanTimeInterval(session_start_time, _now) << "s" << "] " << std::setprecision(20) << message << endl;
}

void PrintFancy(high_resolution_clock::time_point session_start_time, const float &message, const int width, const int precision) {

  high_resolution_clock::time_point _now = high_resolution_clock::now();
  time_t now = chrono::system_clock::to_time_t(_now);
  char str[80];
  strftime(str, sizeof(str), "%d %b %Y %H:%M:%S", localtime(&now));

  cout << "[" << str << " - " << std::setw(width) << std::setprecision(precision) << getHumanTimeInterval(session_start_time, _now) << "s" << "] " << std::setprecision(precision) << message << endl;
}

template <typename T>
void PrintFancy(const T& message) {

  high_resolution_clock::time_point _now = high_resolution_clock::now();
  time_t now = chrono::system_clock::to_time_t(_now);
  char str[80];
  strftime(str, sizeof(str), "%d %b %Y %H:%M:%S", localtime(&now));

  cout << "[" << str << "] " << std::setprecision(20) << message << endl;
}

void PrintWithDelimiters(high_resolution_clock::time_point session_start_time, const string& message) {
  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(session_start_time, message);
  PrintDelimiter(1, 1, 80, '=');
}

void PrintWithDelimiters(const string& message) {
  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(message);
  PrintDelimiter(1, 1, 80, '=');
}

void PrintTimeElapsedSince(high_resolution_clock::time_point start_time, const string& message){
  high_resolution_clock::time_point now = high_resolution_clock::now();
  PrintFancy(start_time, message + " :: " + to_string((chrono::duration_cast<chrono::duration<double> >(now - start_time)).count()) + " seconds");
}

string StringPadding(string original, int charCount ) {
  if (charCount > original.size()) {
    original.resize(charCount, ' ');
  }
  return original;
}