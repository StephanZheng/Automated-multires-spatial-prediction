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

#include "core/DataBlob.h"

inline int MatrixBlob::SerialIndex(int row, int column) {
  if (rows <= 0 || columns <= 0) cout << "row: " << row << "/" << rows << " col: " << column << "/" << columns << " index: " << row * columns + column << endl;
  return row * columns + column;
}
float MatrixBlob::at(int row, int column) {
  CheckCoordinatesAreLegal(row, column);
  return data[SerialIndex(row, column)];
}
float* MatrixBlob::att(int row, int column) {
  CheckCoordinatesAreLegal(row, column);
  return &data[SerialIndex(row, column)];
}
bool MatrixBlob::CheckCoordinatesAreLegal(int row, int column) {
  if (row >= rows) {
    showParameters(row, column);
    assert( row < rows );
    return false;
  }
  else if (column >= columns) {
    showParameters(row, column);
    assert(column < columns);
    return false;
  }
  else {
    return true;
  }
}
void MatrixBlob::init(int _rows, int _columns) {
  data.resize(_rows * _columns);
  rows   = _rows;
  columns  = _columns;
}
void MatrixBlob::showParameters(int row, int column) {
  cout << "Dimensions of MatrixBlob:" << endl;
  cout << row << " " << rows << endl;
  cout << column << " " << columns << endl;
}
void MatrixBlob::showContents(int limit, int modulo) {
  for (int i = 0; i < data.size(); ++i) {
    if (i < limit and i % modulo == 0) {
      cout << setprecision(20) << "Index " << i << " : " << data[i] << endl;
    }
    if (i > limit) {
      break;
    }
  }
}
void MatrixBlob::showMatrixContents(int row_lo, int row_hi, int col_lo, int col_hi, int row_modulo) {

  for (int i = row_lo; i < row_hi; ++i) {

    if (i % row_modulo == 0) {

      cout << setw(7);
      cout << "Row " << i << " ";

      for (int j = col_lo; j < col_hi; ++j) {
        cout << setw(7) << setprecision(5) << this->at(i, j) << " ";
      }

      cout << endl;
    }

  }
}
float MatrixBlob::getMaximum(int begin, int range) {
  float max;
  int pos;
  int i;
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < columns - 2; ++col)
    {
      i = SerialIndex(row, col);
      if (data[i] > max) {
        max = data[i];
        pos = i;
      }
    }
  }

  // for (int i = begin; i < range; ++i) {
  //   if (data[i] > max) {
  //   max = data[i];
  //   pos = i;
  //   }
  // }
  cout << "MatrixBlob max = " << max << " @ " << pos << endl;
  return max;
}
float MatrixBlob::getMinimum(int begin, int range) {
  float min;
  int pos;
  int i;
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < columns - 2; ++col)
    {
      i = SerialIndex(row, col);
      if (data[i] < min) {
        min = data[i];
        pos = i;
      }
    }
  }
  // for (int i = begin; i < range; ++i) {
  //   if (data[i] < min) {
  //   min = data[i];
  //   pos = i;
  //   }
  // }
  cout << "MatrixBlob min = " << min << " @ " << pos << endl;
  return min;
}
void MatrixBlob::rescale(float factor, int begin, int range) {
  int i;
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < columns - 2; ++col)
    {
      i = SerialIndex(row, col);
      data[i] = data[i] * factor;
    }
  }
  // for (int i = begin; i < range; ++i) {
  //   data[i] = data[i] * factor;
  // }
}
void MatrixBlob::rescaleTr(float factor, int begin, int range) {
  int i;
  for (int row = 0; row < rows - 2; ++row)
  {
    for (int col = 0; col < columns; ++col)
    {
      i = SerialIndex(row, col);
      data[i] = data[i] * factor;
    }
  }
  // for (int i = begin; i < range; ++i) {
  //   data[i] = data[i] * factor;
  // }
}
void MatrixBlob::ThresholdValues(float threshold_lower, float threshold_upper, float new_value) {
  // cout << "DEBUG -- Thresholding this data-structure" << endl;
  for (int i = 0; i < data.size(); ++i)
  {
    if (data[i] > threshold_lower and data[i] < threshold_upper ) data[i] = new_value;
  }
}
void MatrixBlob::ClampHardValues(float threshold_lower, float new_value) {
  // cout << "Hard clamping data-structure" << endl;
  for (int i = 0; i < data.size(); ++i)
  {
    if (data[i] < threshold_lower) data[i] = new_value;
  }
}
void MatrixBlob::erase() {
  for (int i; i < data.size(); i++) {
    data[i] = 0.;
  }
}
void MatrixBlob::initRandom(float range) {
  random_device rd;
  mt19937 e2(rd());
  uniform_real_distribution<> u(-range, range);

  for (int i = 0; i < data.size(); ++i)
  {
    data[i] = u(e2);
  }
}