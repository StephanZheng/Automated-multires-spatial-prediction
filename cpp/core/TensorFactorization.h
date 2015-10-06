

#ifndef TENSORFACTORIZATION_H
#define TENSORFACTORIZATION_H

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

#include "config/GlobalConstants.h"
#include "config/Settings.h"

#include "core/Trainer.h"
#include "core/DataBlob.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"

#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"

using namespace std;

class StageTwoController : public Trainer {
  // This class calls a Python routine to do non-negative tensor factorization + upscaling of computed weight matrices
  public:

  Tensor3Blob Weight_U;
  MatrixBlob  FactorA;
  MatrixBlob  FactorB;
  MatrixBlob  FactorC;
  MatrixBlob  LatentPsi;
  Tensor3Blob WeightW;

  StageTwoController(IOController *aIOController_,
            MatrixBlob *aBlob_B_,
            MatrixBlob *aBlob_BTransposed_,
            MatrixBlob *aBlob_C_,
            MatrixBlob *aBlob_CTransposed_,
            GroundTruthLabel *aGroundTruthLabelsTrainValStrong_,
            GroundTruthLabel *aGroundTruthLabelsTestStrong_,
            GroundTruthLabel *aGroundTruthLabelsTrainValWeak_,
            GroundTruthLabel *aGroundTruthLabelsTestWeak_,
            Settings *aSettings_,
            CurrentStateBlob *aCurrentStateBlob_){

    // Tell ThreeMatrixFactorTrainer where the loaded data sits
    IOController_          = aIOController_;
    Blob_B_              = aBlob_B_;
    Blob_BTransposed_        = aBlob_BTransposed_;
    Blob_C_              = aBlob_C_;
    Blob_CTransposed_        = aBlob_CTransposed_;
    GroundTruthLabelsTrainValStrong_ = aGroundTruthLabelsTrainValStrong_;
    GroundTruthLabelsTestStrong_   = aGroundTruthLabelsTestStrong_;
    GroundTruthLabelsTrainValWeak_   = aGroundTruthLabelsTrainValWeak_;
    GroundTruthLabelsTestWeak_     = aGroundTruthLabelsTestWeak_;
    Settings_            = aSettings_;
    CurrentStateBlob_        = aCurrentStateBlob_;

    int n_dimension_B        = Settings_->StageOne_Dimension_B;
    int n_dimension_C        = Settings_->StageOne_Dimension_C;
    int n_frames           = Settings_->NumberOfFrames;
    int n_latent_dimensions      = Settings_->NumberOfLatentDimensions;
    int n_dimension_A        = Settings_->Dimension_A;
    int n_threads          = Settings_->StageOne_NumberOfThreads;

    Weight_U.init(n_dimension_A, n_dimension_B, n_dimension_C);
    WeightW.init(n_dimension_B, n_dimension_C, n_dimension_A);

  }

  void  NonNegativeTensorFactorization(Tensor3Blob * tensor, MatrixBlob * FactorA, MatrixBlob * FactorB, MatrixBlob * FactorC);
  float   ComputeTripleProductABC(MatrixBlob * FactorA, MatrixBlob * FactorB, MatrixBlob * FactorC);
  float   TensorEuclideanNorm(Tensor3Blob * TensorA, Tensor3Blob * TensorB);
  void  UpdateFactor(MatrixBlob * FactorA, MatrixBlob * FactorB, MatrixBlob * FactorC, int factor);

  void  Load_SnapshotWeights(string weights_snapshot_file, string momentum_snapshot_file);
  void  Store_Snapshot(int cross_val_run, int epoch, string fp_snapshot);
};

#endif