

#include <boost/chrono.hpp>
#include <boost/timer.hpp>
#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // isnan
#include <chrono>
#include <random>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <queue>
#include <assert.h>
#include <sstream>
#include <fstream>

#include "util/IOController.h"

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
// void WriteToFile(string filename, char* value) {
//   WriteToFile<char>(filename, value);
// }

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

void WriteNewlineToFile(string filename) {
  std::ofstream LogFile(filename, ios::app);
  if (LogFile.is_open())
  {
    LogFile << endl;
    LogFile.close();
  }
  else {
    PrintFancy() << "Unable to open file " << filename << endl;
  }
}

void writeBlobToFile(FILE * fp, VectorBlob * v) {
  for (int i = 0; i < v->data.size(); ++i) fwrite(&v->data[i], sizeof(float), 1, fp);
}
void writeBlobToFile(FILE * fp, MatrixBlob * m) {
  for (int i = 0; i < m->data.size(); ++i) fwrite(&m->data[i], sizeof(float), 1, fp);
}
void writeBlobToFile(FILE * fp, Tensor3Blob * t) {
  for (int i = 0; i < t->data.size(); ++i) fwrite(&t->data[i], sizeof(float), 1, fp);
}


void IOController::LoadFeaturesFromFile(string fp_features, MatrixBlob *blob, int loadtranspose, int which_feature, int stage) {

  PrintFancy(Settings_->session_start_time, "This is LoadFeaturesFromFile");

  FILE  * pFile;
  int   * frame_id      = new int;
  int   * ball_handler_pos  = new int;
  int   * buffer      = new int;
  size_t  result;

  int n_integers_in_file   = 0;
  int n_integers_per_frame = 0;

  PrintFancy(Settings_->session_start_time, "Loading occupancy features from: "+fp_features);
  pFile = fopen(fp_features.c_str(), "rb");
  if (pFile == NULL) PrintFancy(Settings_->session_start_time, " was null");

  while ((result = fread(buffer, sizeof(*buffer), 1, pFile))) n_integers_in_file++;

  PrintFancy(Settings_->session_start_time, "Read-in n_integers_in_file   : "+to_string(n_integers_in_file));

  if (stage == 1){
    if (which_feature == 0) {
      n_integers_per_frame = Settings_->StageOne_NumberOfNonZeroEntries_B;
    } else if (which_feature == 1) {
      n_integers_per_frame = Settings_->DummyMultiplier * Settings_->StageOne_NumberOfNonZeroEntries_C;
    } else {
      PrintFancy(Settings_->session_start_time, "You didn't give the right option!");
      return;
    }
  } else if (stage == 3) {
    if (which_feature == 0) {
      n_integers_per_frame = Settings_->StageThree_NumberOfNonZeroEntries_B;
    } else if (which_feature == 1) {
      n_integers_per_frame = Settings_->DummyMultiplier * (Settings_->StageThree_NumberOfNonZeroEntries_C);
    } else if (which_feature == 2) {
      n_integers_per_frame = Settings_->StageThree_NumberOfNonZeroEntries_B * NUMBER_OF_INTS_WINDOW_3BY3;
    } else if (which_feature == 3) {
      n_integers_per_frame = Settings_->DummyMultiplier * ((Settings_->StageThree_NumberOfNonZeroEntries_C - Settings_->Dimension_C_BiasSlice) * NUMBER_OF_INTS_WINDOW_3BY3);

      cout << "Def SlidingWindow file, I expect " << n_integers_per_frame * blob->rows << " " << Settings_->StageThree_NumberOfNonZeroEntries_C << " " << Settings_->Dimension_C_BiasSlice << " " << (Settings_->StageThree_NumberOfNonZeroEntries_C - Settings_->Dimension_C_BiasSlice) * NUMBER_OF_INTS_WINDOW_3BY3 << " " << n_integers_per_frame << endl;
    } else {
      PrintFancy(Settings_->session_start_time, "You didn't give the right option!");
      return;
    }
  }

  PrintFancy(Settings_->session_start_time, "Check: n_integers_in_file   " + to_string(n_integers_in_file)  +" =?= "+ to_string(n_integers_per_frame * blob->rows) + "blob->rows * blob->columns");
  PrintFancy(Settings_->session_start_time, "Check: n_integers_per_frame " + to_string(n_integers_per_frame) +" =?= "+ to_string(blob->columns) + " blob->columns");

  if (loadtranspose == 0) assert(n_integers_in_file == n_integers_per_frame * blob->rows);
  else if (loadtranspose == 1) assert(n_integers_in_file == n_integers_per_frame * blob->columns);
  else return;

  assert(n_integers_in_file % n_integers_per_frame == 0);

  PrintFancy(Settings_->session_start_time, "rows: "+to_string(blob->rows) + " columns: "+to_string(blob->columns));

  fseek(pFile, 0, SEEK_SET);

  *frame_id = 0;

  for (int k = 0; k < n_integers_in_file / n_integers_per_frame; k++) {
    // result = fread(frame_id, sizeof(*frame_id), 1, pFile);

    if (loadtranspose == 0) assert(blob->SerialIndex(*frame_id, 0) < blob->data.size());
    else if (loadtranspose == 1) assert(blob->SerialIndex(0, *frame_id) < blob->data.size());
    else return;

    for (int i = 0; i < n_integers_per_frame; ++i) // Frame_id is read separately
    {
      result = fread(ball_handler_pos, sizeof(*ball_handler_pos), 1, pFile);
      // assert(*ball_handler_pos >= 0);

      if (loadtranspose == 0) {
        blob->data[blob->SerialIndex(*frame_id, i)] = *ball_handler_pos;
      }
      else if (loadtranspose == 1) {
        blob->data[blob->SerialIndex(i, *frame_id)] = *ball_handler_pos;
      }
      else {
        PrintFancy(Settings_->session_start_time, "You didn't give the right option!");
        return;
      }

      // if (k % 5000000 == 0) cout << *frame_id << " read: " << *ball_handler_pos << endl;
    }
    *frame_id += 1;
  }
  if (loadtranspose == 0) {
    if (which_feature == 0) PrintFancy(Settings_->session_start_time, "Ballhandler features loaded");
    if (which_feature == 1) PrintFancy(Settings_->session_start_time, "Defender features loaded");
    if (which_feature == 2) PrintFancy(Settings_->session_start_time, "Ballhandler sliding window 3x3 features loaded");
    if (which_feature == 3) PrintFancy(Settings_->session_start_time, "Defender sliding window 3x3 features loaded");
    PrintFancy(Settings_->session_start_time, "FeatureBlob_ has been loaded into row-first MatrixDataBlob of size "+to_string(blob->data.size()));
  } else if (loadtranspose == 1) {
    PrintFancy(Settings_->session_start_time, "FeatureBlob (transposed) has been loaded into row-first MatrixDataBlob of size "+to_string(blob->data.size()));
  } else return;
  fclose(pFile);


  delete frame_id;
  delete ball_handler_pos;
  delete buffer;
}

void IOController::LoadGroundTruthLabelsStrong(string fp_ground_truth_labels, GroundTruthLabel *blob, int sanity_check) {

  FILE  * fp;
  int   * buffer = new int;
  size_t  result;

  PrintFancy(Settings_->session_start_time, "Loading ground truth labels : "+fp_ground_truth_labels);

  fp   = fopen(fp_ground_truth_labels.c_str(), "rb");
  if (fp == NULL) PrintFancy(Settings_->session_start_time, " was null");

  // Find how many entries there are in the file, resize data-holder accordingly
  int n_integers_in_file = 0;
  while ((result = fread(buffer, sizeof(*buffer), 1, fp))) n_integers_in_file++;

  PrintFancy(Settings_->session_start_time, "Read-in n_integers_in_file  : "+to_string(n_integers_in_file));

  // Check that the number of strong labels loaded corresponds to the number of strong labels defined in
  // our settings class
  Settings *Settings_ = this->Settings_;
  assert(n_integers_in_file % 3 == 0);

  int n_label_tuples = n_integers_in_file / 3;
  assert(n_label_tuples == sanity_check);

  fseek(fp, 0, SEEK_SET);

  for (int k = 0; k < n_label_tuples; k++) {
    fread(buffer, sizeof(*buffer), 1, fp);
    blob[k].index_A     = *buffer;
    fread(buffer, sizeof(*buffer), 1, fp);
    blob[k].frame_id = *buffer;
    fread(buffer, sizeof(*buffer), 1, fp);
    blob[k].ground_truth_label     = *buffer;
    // cout << k << " StrongLabel Action: " << blob[k].index_A << " Frame " << blob[k].frame_id << " Present? " << blob[k].ground_truth_label << endl;
  }
  PrintFancy(Settings_->session_start_time, "GroundTruthLabel loaded with: "+to_string(n_label_tuples) + " strong labels");
  fclose(fp);

  delete buffer;
}
void IOController::LoadGroundTruthLabelsWeak(string fp_ground_truth_labels, GroundTruthLabel *blob, int sanity_check) {

  FILE  * fp;
  int   * buffer = new int;
  size_t  result;

  PrintFancy(Settings_->session_start_time, "Loading ground truth labels : "+fp_ground_truth_labels);

  fp   = fopen(fp_ground_truth_labels.c_str(), "rb");
  if (fp == NULL) PrintFancy(Settings_->session_start_time, " was null");

  // Find how many entries there are in the file, resize data-holder accordingly
  int n_integers_in_file = 0;
  while ((result = fread(buffer, sizeof(*buffer), 1, fp))) n_integers_in_file++;

  PrintFancy(Settings_->session_start_time, "Read-in n_integers_in_file  : "+to_string(n_integers_in_file));

  // Check that the number of strong labels loaded corresponds to the number of strong labels defined in
  // our settings class
  Settings *Settings_ = this->Settings_;
  int n_label_tuples = n_integers_in_file / 3;
  assert(n_integers_in_file % 3 == 0);
  assert(n_label_tuples == sanity_check);

  fseek(fp, 0, SEEK_SET);

  for (int k = 0; k < n_label_tuples; k++) {
    fread(buffer, sizeof(*buffer), 1, fp);
    blob[k].index_A     = *buffer;
    fread(buffer, sizeof(*buffer), 1, fp);
    blob[k].frame_id = *buffer;
    fread(buffer, sizeof(*buffer), 1, fp);
    assert(*buffer == 0);
    blob[k].ground_truth_label   = *buffer;
    // cout << k << " WeakLabel Action: " << blob[k].index_A << " Frame " << blob[k].frame_id << " Present? " << blob[k].ground_truth_label << endl;
  }
  PrintFancy(Settings_->session_start_time, "GroundTruthLabel loaded with: "+to_string(n_label_tuples) + " weak labels");
  fclose(fp);

  delete buffer;
}