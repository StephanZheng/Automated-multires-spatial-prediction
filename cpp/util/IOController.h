// Copyright 2014 Stephan Zheng

#ifndef IOCONTROLLER_H
#define IOCONTROLLER_H

#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <chrono>
#include <time.h>
#include <random>
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

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/reader.h"

#include "config/GlobalConstants.h"
#include "config/Settings.h"
#include "core/DataBlob.h"
#include "util/CurrentStateBlob.h"
#include "util/PrettyOutput.h"

using namespace std;

class IOController {
  public:
  string name;
  Settings      *Settings_;
  CurrentStateBlob  *CurrentStateBlob_;

  explicit IOController(Settings *aSettings_, CurrentStateBlob *aCurrentStateBlob_) {
    Settings_ = aSettings_;
    CurrentStateBlob_ = aCurrentStateBlob_;
  }

  void LoadFeaturesFromFile(string fp_features, MatrixBlob *blob, int loadtranspose, int bh_or_def, int stage);
  void LoadGroundTruthLabelsStrong(string fp_ground_truth_labels, GroundTruthLabel *blob, int sanity_check);
  void LoadGroundTruthLabelsWeak(string fp_ground_truth_labels, GroundTruthLabel *blob, int sanity_check);

  void ExportFinalResults();

  void OpenNewFile(string filename) {
    std::ofstream LogFile(filename, ios::out);
    LogFile << "";
    LogFile.close();
  }

  void WriteToFile(string filename, float value) {
    WriteToFile<float>(filename, value);
  }
  void WriteToFile(string filename, double value) {
    WriteToFile<double>(filename, value);
  }
  void WriteToFile(string filename, int value) {
    WriteToFile<int>(filename, value);
  }
  void WriteToFile(string filename, string value) {
    WriteToFile<string>(filename, value);
  }

  template <typename T>
  void WriteToFile(string filename, T value) {
    std::ofstream LogFile(filename, ios::app);
    if (LogFile.is_open())
    {
      LogFile << value;
      LogFile.close();
    }
    else {
      PrintFancy() << "Unable to open file " << filename << endl;
    }
  }

  // std::ofstream AppendToFile(string filename) {
  //   std::ofstream File(filename, ios::app);
  //   return File;
  // }

};

#endif