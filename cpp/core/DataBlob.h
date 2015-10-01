// Copyright 2014 Stephan Zheng

#ifndef DATABLOB_H
#define DATABLOB_H

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
#include <iostream>
#include <iomanip>
#include <fstream>

#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "config/GlobalConstants.h"
#include "util/PrettyOutput.h"
#include "test/Test.h"

using namespace std;

class DataBlob {
public:
  string name;
  explicit DataBlob() {}
};

class VectorBlob : public DataBlob {
public:
  int columns;
  boost::numeric::ublas::vector<float> data;
  string name;

  VectorBlob() {}

  explicit VectorBlob(int size) : data(size) {}

  inline float at(int column) {
    assert( column < data.size() );
    return data[column];
  }
  float * att(int column) {
    assert( column < data.size() );
    return &data[column];
  }
  inline float last() {
    if (data.size() > 0) {
      return data[data.size() - 1];
    } else {
      return 0.;
    }
  }
  void showContents(int limit, int modulo) {
    for (int i = 0; i < data.size(); ++i) {
      if (i < limit and i % modulo == 0) {
        cout << setprecision(20) << "Index " << i << " : " << data[i] << endl;
      }
      if (i > limit) {
        break;
      }
    }
  }
  void init(int _columns) {
    data.resize(_columns);
    columns = _columns;
  }
  void erase() {
    for (int i; i < data.size(); i++) {
      data[i] = 0.;
    }
  }
  void push_back(float x) {

    assert (columns == data.size());
    data.resize(columns + 1);
    columns += 1;
    assert (columns == data.size());
    data[data.size() - 1] = x;

  }
  void showVectorContents(int limit, int linebreak) {
    assert (linebreak >= 1);
    for (int i = 0; i < columns; ++i) {
      if (i >= limit) break;

      cout << setw(7);
      if (i % linebreak == 0) cout << "Col " << i << " ";
      cout << setw(14) << setprecision(8) << this->at(i) << " ";
      if ((i + 1) % linebreak == 0) cout << endl;
    }
    cout << endl;
  }
  void initRandom(float range) {
    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> u(-range, range);

    for (int i = 0; i < data.size(); ++i)
    {
      data[i] = u(e2);
    }
  }
};
/**
 * Note: Matrices are serialized row-major
 */
class MatrixBlob : public DataBlob {
public:

  int rows;
  int columns;

  string name;
  boost::numeric::ublas::vector<float> data;

  VectorBlob sp_sd1_lvls;
  VectorBlob sp_sd1_nonzeros;
  VectorBlob sp_sd2_lvls;
  VectorBlob sp_sd2_nonzeros;

  MatrixBlob() {
    sp_sd1_lvls.init(0);
    sp_sd1_nonzeros.init(0);
    sp_sd2_lvls.init(0);
    sp_sd2_nonzeros.init(0);
  }
  MatrixBlob(int _rows, int _columns) {
    rows  = _rows;
    columns = _columns;
    data.resize(rows * columns);
    // showParameters(0, 0);
    sp_sd1_lvls.init(0);
    sp_sd1_nonzeros.init(0);
    sp_sd2_lvls.init(0);
    sp_sd2_nonzeros.init(0);
  }
  inline int SerialIndex(int row, int column);
  float at(int row, int column);
  float *att(int row, int column);
  bool CheckCoordinatesAreLegal(int row, int column);
  void init(int _rows, int _columns);
  void showParameters(int row, int column);
  void showContents(int limit, int modulo);
  void showMatrixContents(int row_lo, int row_hi, int col_lo, int col_hi, int row_modulo);
  float getMaximum(int begin, int range);
  float getMinimum(int begin, int range);
  void rescale(float factor, int begin, int range);
  void rescaleTr(float factor, int begin, int range);
  void ThresholdValues(float threshold_lower, float threshold_upper, float new_value);
  void ClampHardValues(float threshold_lower, float new_value);
  void erase();
  void initRandom(float range);
};

class Tensor3Blob : public DataBlob {
public:
  int slices;
  int rows;
  int columns;
  boost::numeric::ublas::vector<float> data;
  Tensor3Blob() {}
  Tensor3Blob(int _slices, int _rows, int _columns) {
    slices  = _slices;
    rows  = _rows;
    columns = _columns;
    data.resize(slices * rows * columns);
  }
  inline int SerialIndex(int slice, int row, int col) {
    if (slices <= 0 || rows <= 0 || columns <= 0) cout << "slice: " << slice << "/" << slices << " col: " << row << "/" << rows << " index: " << slice * rows * columns + row * columns + col << endl;
    return slice * rows * columns + row * columns + col;
  }
  inline float at(int slice, int row, int col) {
    CheckCoordinatesAreLegal(row, col, slice);
    int index = SerialIndex(slice, row, col);
    assert(index < slices * rows * columns);
    return data[index];
  }

  float *att(int slice, int row, int col) {
    CheckCoordinatesAreLegal(row, col, slice);
    return &data[SerialIndex(slice, row, col)];
  }

  bool CheckCoordinatesAreLegal(int row, int column, int slice) {
    if (row >= rows) {
      showParameters(row, column, slice);
      assert( row < rows );
      return false;
    }
    else if (column >= columns) {
      showParameters(row, column, slice);
      assert(column < columns);
      return false;
    }
    else if (slice >= slices) {
      showParameters(row, column, slice);
      assert(slice < slices);
      return false;
    }
    else {
      return true;
    }
  }

  void init(int _slices, int _rows, int _columns) {
    data.resize(_slices * _rows * _columns);
    slices = _slices;
    rows   = _rows;
    columns   = _columns;
  }

  void erase() {
    for (int i; i < data.size(); i++) {
      data[i] = 0.;
    }
  }

  void showParameters(int row, int column, int slice) {
    cout << "Dimensions of TensorBlob:" << endl;
    cout << row << " " << rows << endl;
    cout << column << " " << columns << endl;
    cout << slice << " " << slices << endl;
  }

  void showContents(int limit, int modulo) {
    for (int i = 0; i < data.size(); ++i) {
      if (i < limit and i % modulo == 0) {
        cout << setprecision(20) << "Index " << i << " : " << data[i] << endl;
      }
      if (i > limit) {
        break;
      }
    }
  }

  void showTensorContents(int row_limit, int row_modulo, int col_limit, int col_modulo, int slice) {

    for (int i = 0; i < rows; ++i) {

      if (i > row_limit) break;

      if (i <= row_limit and i % row_modulo == 0) {

        cout << setw(7);
        cout << "slice " << slice << " row " << i <<  " " ;

        for (int j = 0; j < columns; ++j) {

          if (j > col_limit) break;

          if (j < col_limit and j % col_modulo == 0) {
            cout << setw(14) << setprecision(8) << this->at(i, j, slice) << " ";
          }

        }

      }
      cout << endl;

    }
  }

  void initRandom(float range) {
    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> u(-range, range);

    for (int i = 0; i < data.size(); ++i)
    {
      data[i] = u(e2);
    }
  }

  void ThresholdValues(float threshold_lower, float threshold_upper, float new_value) {
    for (int i = 0; i < data.size(); ++i)
    {
      if (data[i] > threshold_lower and data[i] < threshold_upper ) data[i] = new_value;
    }
  }

  // float getMaximum(int begin, int range) {
  //   float max;
  //   int pos;
  //   int i;
  //   for (int slice = 0; slice < slices; ++slice)
  //   {
  //   for (int col = 0; col < rows - 2; ++col)
  //   {
  //   i = SerialIndex(slice, col, col);
  //   if (data[i] > max) {
  //   max = data[i];
  //   pos = i;
  //   }
  //   }
  //   }

  //   // for (int i = begin; i < range; ++i) {
  //   //  if (data[i] > max) {
  //   //  max = data[i];
  //   //  pos = i;
  //   //  }
  //   // }
  //   cout << "MatrixBlob max = " << max << " @ " << pos << endl;
  //   return max;
  // }
  // float getMinimum(int begin, int range) {
  //   float min;
  //   int pos;
  //   int i;
  //   for (int slice = 0; slice < slices; ++slice)
  //   {
  //   for (int col = 0; col < rows - 2; ++col)
  //   {
  //   i = SerialIndex(slice, col);
  //   if (data[i] < min) {
  //   min = data[i];
  //   pos = i;
  //   }
  //   }
  //   }
  //   // for (int i = begin; i < range; ++i) {
  //   //  if (data[i] < min) {
  //   //  min = data[i];
  //   //  pos = i;
  //   //  }
  //   // }
  //   cout << "MatrixBlob min = " << min << " @ " << pos << endl;
  //   return min;
  // }
  // void rescale(float factor, int begin, int range) {
  //   int i;
  //   for (int slice = 0; slice < slices; ++slice)
  //   {
  //   for (int col = 0; col < rows - 2; ++col)
  //   {
  //   i = SerialIndex(slice, col);
  //   data[i] = data[i] * factor;
  //   }
  //   }
  //   // for (int i = begin; i < range; ++i) {
  //   //  data[i] = data[i] * factor;
  //   // }
  // }
  // void rescaleTr(float factor, int begin, int range) {
  //   int i;
  //   for (int slice = 0; slice < slices - 2; ++slice)
  //   {
  //   for (int col = 0; col < rows; ++col)
  //   {
  //   i = SerialIndex(slice, col);
  //   data[i] = data[i] * factor;
  //   }
  //   }
  //   // for (int i = begin; i < range; ++i) {
  //   //  data[i] = data[i] * factor;
  //   // }
  // }
  // void ThresholdValues(float threshold, int begin, int range) {
  //   int i;
  //   for (int slice = 0; slice < slices; ++slice)
  //   {
  //   for (int col = 0; col < rows - 2; ++col)
  //   {
  //   i = SerialIndex(slice, col);
  //   if (data[i] < threshold) data[i] = 0.;
  //   }
  //   }
  //   // for (int i = begin; i < range; ++i)
  //   // {
  //   //  if (data[i] < threshold) data[i] = 0.;
  //   // }
  // }
};



using namespace boost::numeric::ublas;

class SparseTensor3Blob : public DataBlob {
public:
  int slices;
  int rows;
  int columns;

  compressed_matrix<float, column_major> data;

  SparseTensor3Blob() {}
  SparseTensor3Blob(int _slices, int _rows, int _columns) {
    slices  = _slices;
    rows  = _rows;
    columns = _columns;
    data.resize(slices, rows * columns);
  }

  inline float at(int slice, int row, int col) {
    assert( slice < slices );
    assert( row < rows );
    assert( col < columns );
    return data(slice, row * columns + col);
  }

  inline int SerialIndex(int row, int column) {
    if (rows <= 0 || columns <= 0) cout << "row: " << row << "/" << rows << " col: " << column << "/" << columns << " index: " << row * columns + column << endl;
    return row * columns + column;
  }

  // float *att(int slice, int row, int col) {
  //   assert( slice < slices );
  //   assert( row < rows );
  //   assert( col < columns );
  //   return &data(slice, row * columns + col);
  // }

  void init(int _slices, int _rows, int _columns) {
    data.resize(_slices, _rows * _columns);
    slices  = _slices;
    rows  = _rows;
    columns = _columns;
  }

  void erase() {
    data.clear();
  }
};

class GroundTruthLabel {
public:
  int index_A;
  int frame_id;
  int ground_truth_label;
};

#endif