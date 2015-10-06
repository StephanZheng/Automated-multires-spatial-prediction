

#ifndef THREEMATRIXFACTORTRAINER_H
#define THREEMATRIXFACTORTRAINER_H

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

#include "core/Trainer.h"
#include "core/DataBlob.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"

#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"
#include "util/TrainUtil.h"

using namespace std;

class ThreeMatrixFactorTrainer : public Trainer {

  public:
  // These are parameters to be learned
  // Legacy
  Tensor3Blob WeightW;
  Tensor3Blob WeightW_transposed;
  Tensor3Blob WeightW_Diff;
  Tensor3Blob WeightW_DiffSq_Cache;

  // X
  MatrixBlob LF_A;
  MatrixBlob LF_A_Diff;
  MatrixBlob LF_A_DiffSq_Cache;
  MatrixBlob LF_A_Snapshot;

  // Y
  MatrixBlob LF_B;
  MatrixBlob LF_B_Diff;
  MatrixBlob LF_B_DiffSq_Cache;
  MatrixBlob LF_B_Snapshot;

  MatrixBlob LF_B_updated_bool_check;

  // Sparse latent factor
  MatrixBlob SLF_B;
  MatrixBlob SLF_B_Diff;
  MatrixBlob SLF_B_DiffSq_Cache;
  MatrixBlob SLF_B_Snapshot;

  // Z
  MatrixBlob LF_C;
  MatrixBlob LF_C_Diff;
  MatrixBlob LF_C_DiffSq_Cache;
  MatrixBlob LF_C_Snapshot;

  Tensor3Blob Sparse_S;
  Tensor3Blob Sparse_S_Diff;
  Tensor3Blob Sparse_S_DiffSq_Cache;
  Tensor3Blob Sparse_S_Snapshot;

  VectorBlob Bias_A;
  VectorBlob Bias_A_Diff;
  VectorBlob Bias_A_DiffSq_Cache;
  VectorBlob Bias_A_Snapshot;

  // Learning hyperparameters
  float regularization_W;
  float regularization_Bias_A;
  float regularization_LF_A;
  float regularization_Sparse_S;

  int bh_outofbounds_counter;

  ThreeMatrixFactorTrainer(IOController *aIOController_,
            MatrixBlob *aBlob_B_,
            MatrixBlob *aBlob_BB_,
            MatrixBlob *aBlob_BTransposed_,
            MatrixBlob *aBlob_C_,
            MatrixBlob *aBlob_CC_,
            MatrixBlob *aBlob_CTransposed_,
            GroundTruthLabel *aGroundTruthLabelsTrainValStrong_,
            GroundTruthLabel *aGroundTruthLabelsTestStrong_,
            GroundTruthLabel *aGroundTruthLabelsTrainValWeak_,
            GroundTruthLabel *aGroundTruthLabelsTestWeak_,
            Settings *aSettings_,
            CurrentStateBlob *aCurrentStateBlob_) : \
          Trainer(aSettings_->StageThree_Dimension_B,
                  aSettings_->SpatialEntropy_NumberOfBins,
                  aSettings_->SpatialEntropy_MinValue,
                  aSettings_->SpatialEntropy_MaxValue) {

    // Temp
    bh_outofbounds_counter           = 0;

    // Tell ThreeMatrixFactorTrainer where the loaded data sits
    IOController_                    = aIOController_;
    Blob_B_                          = aBlob_B_;
    Blob_BB_                         = aBlob_BB_;
    Blob_BTransposed_                = aBlob_BTransposed_;
    Blob_C_                          = aBlob_C_;
    Blob_CC_                         = aBlob_CC_;
    Blob_CTransposed_                = aBlob_CTransposed_;
    GroundTruthLabelsTrainValStrong_ = aGroundTruthLabelsTrainValStrong_;
    GroundTruthLabelsTestStrong_     = aGroundTruthLabelsTestStrong_;
    GroundTruthLabelsTrainValWeak_   = aGroundTruthLabelsTrainValWeak_;
    GroundTruthLabelsTestWeak_       = aGroundTruthLabelsTestWeak_;
    Settings_                        = aSettings_;
    CurrentStateBlob_                = aCurrentStateBlob_;

    int n_dimension_A                = Settings_->Dimension_A;
    int n_dimension_B                = Settings_->StageThree_Dimension_B;
    int n_dimension_C                = Settings_->StageThree_Dimension_C;
    int n_latent_dimensions          = Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2;
    int n_frames                     = Settings_->NumberOfFrames;
    int n_threads                    = Settings_->StageThree_NumberOfThreads;

    assert (Settings_->NumberOfLatentDimensions == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);


    // VectorBlob *ptr_v;
    // MatrixBlob *ptr_m;

    ConditionalScore.init(    n_frames, Settings_->NumberOfColumnsScoreFunction);

    LF_A.init(          n_dimension_A, n_latent_dimensions);
    LF_A_Diff.init(       n_dimension_A, n_latent_dimensions);
    LF_A_DiffSq_Cache.init(   n_dimension_A, n_latent_dimensions);
    LF_A_Snapshot.init(     n_dimension_A, n_latent_dimensions);

    LF_A.name = "Latent Factor A";

    LF_B.init(          n_dimension_B, n_latent_dimensions);
    LF_B_Diff.init(       n_dimension_B, n_latent_dimensions);
    LF_B_DiffSq_Cache.init(   n_dimension_B, n_latent_dimensions);
    LF_B_Snapshot.init(     n_dimension_B, n_latent_dimensions);

    LF_B.name = "Latent Factor B";

    LF_B_updated_bool_check.init(n_dimension_B, n_latent_dimensions);

    if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
      SLF_B.init(          n_dimension_B, n_latent_dimensions);
      SLF_B_Diff.init(       n_dimension_B, n_latent_dimensions);
      SLF_B_DiffSq_Cache.init(   n_dimension_B, n_latent_dimensions);
      SLF_B_Snapshot.init(     n_dimension_B, n_latent_dimensions);
    }

    LF_C.init(          n_dimension_C, n_latent_dimensions);
    LF_C_Diff.init(       n_dimension_C, n_latent_dimensions);
    LF_C_DiffSq_Cache.init(   n_dimension_C, n_latent_dimensions);
    LF_C_Snapshot.init(     n_dimension_C, n_latent_dimensions);

    LF_C.name = "Latent Factor C";

    if (Settings_->StageThree_UseSparseMatS == 1) {
      PrintFancy(Settings_->session_start_time, "Initializing Sparse Mat S!");
      Sparse_S.init(         n_dimension_A, n_dimension_B, n_dimension_C);
      Sparse_S_Diff.init(      n_dimension_A, n_dimension_B, n_dimension_C);
      Sparse_S_DiffSq_Cache.init(  n_dimension_A, n_dimension_B, n_dimension_C);
      Sparse_S_Snapshot.init(    n_dimension_A, n_dimension_B, n_dimension_C);
    } else {
      PrintFancy(Settings_->session_start_time, "We don't initialize Sparse Mat S!");
    }

    Bias_A.init(         n_dimension_A);
    Bias_A_Diff.init(      n_dimension_A);
    Bias_A_DiffSq_Cache.init(  n_dimension_A);
    Bias_A_Snapshot.init(    n_dimension_A);

    // Train-val split creation
    // When constructor is called, we shuffle the indices of the strong-weak labels
    // We only randomize the indices ONCE!

    // Every cross-val run, we call the PermuteTrainValIndices method to permute through the train / val indices

    std::vector<int> strong_indices(Settings_->NumberOfStrongLabels_TrainVal);
    std::vector<int> weak_indices(Settings_->NumberOfWeakLabels_TrainVal);
    std::iota(strong_indices.begin(), strong_indices.end(), 0);
    std::iota(weak_indices.begin(), weak_indices.end(), 0);

    mt19937 g(static_cast<uint32_t>(time(0)));
    std::shuffle(strong_indices.begin(), strong_indices.end(), g);
    std::shuffle(weak_indices.begin(), weak_indices.end(), g);

    CurrentTrainIndices_Strong.resize(Settings_->NumberOfStrongLabels_Train);
    CurrentValidIndices_Strong.resize(Settings_->NumberOfStrongLabels_Val);
    CurrentTrainIndices_Weak.resize(Settings_->NumberOfWeakLabels_Train);
    CurrentValidIndices_Weak.resize(Settings_->NumberOfWeakLabels_Val);

    RandomizedIterator.resize(Settings_->NumberOfStrongLabels_Train + Settings_->NumberOfWeakLabels_Train);

    for (int i = 0; i < CurrentTrainIndices_Strong.size(); ++i) CurrentTrainIndices_Strong[i] = strong_indices[i];
    for (int i = 0; i < CurrentValidIndices_Strong.size(); ++i) CurrentValidIndices_Strong[i] = strong_indices[Settings_->NumberOfStrongLabels_Train + i];
    for (int i = 0; i < CurrentTrainIndices_Weak.size(); ++i)   CurrentTrainIndices_Weak[i] = weak_indices[i];
    for (int i = 0; i < CurrentValidIndices_Weak.size(); ++i)   CurrentValidIndices_Weak[i] = weak_indices[Settings_->NumberOfWeakLabels_Train + i];
    for (int i = 0; i < RandomizedIterator.size(); ++i)     RandomizedIterator[i] = i;

    // Momentum update coefficients
    NestorovMomentumLambda.resize(0);
    NestorovMomentumPreviousWeight.resize(0);
    NestorovMomentumPreviousLF_A.resize(0);
    NestorovMomentumPreviousLF_B.resize(0);
    NestorovMomentumPreviousLF_C.resize(0);
    NestorovMomentumPreviousSparse_S.resize(0);
    NestorovMomentumPreviousBias_A.resize(0);
  }


  int DecisionUnit(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test, int * have_recorded_a_trainloss);
  void StoreStartingWeights();
  void RestoreStartingWeights();

  // Routine that contains the stage 1 train logic
  // aka MASTER
  int   TrainStageThree(string fp_snapshot);

  void  ThreadComputer(int thread_id);

  void  InitializeWeightsRandom(float range_W, float range_A, float range_psi, float range_S);

  float   ComputeInnerproduct(int thread_id, int outer_row, int outer_col, int begin_index, int end_index, MatrixBlob *A, MatrixBlob *B);
  float   ComputeInnerproductTriple(int thread_id, int outer_row, int outer_col, MatrixBlob *A, MatrixBlob *B, MatrixBlob *C);

  // Use in update rule for latent factor psi
  float   ComputeProductPsiPsiAB(   int thread_id,
                    int frame_id,
                    int index_A,
                    int latent_index,
                    MatrixBlob *LF_A,
                    MatrixBlob *LF_B,
                    std::vector<int> *indices_B
                    );
  float   ComputeProductPsiPsiAC(   int thread_id,
                    int frame_id,
                    int index_A,
                    int latent_index,
                    MatrixBlob *LF_A,
                    MatrixBlob *LF_C,
                    std::vector<int> *indices_C
                    );
  float   ComputeProductPsiPsiBC(   int thread_id,
                    int frame_id,
                    int latent_index,
                    MatrixBlob *LF_B,
                    MatrixBlob *LF_C,
                    std::vector<int> *indices_B,
                    std::vector<int> *indices_C
                    );

  float   ComputeProductSC(       int thread_id,
                    int frame_id,
                    int index_A,
                    int index_B,
                    int latent_index,
                    Tensor3Blob *Sparse_S,
                    MatrixBlob *LF_A,
                    MatrixBlob *LF_B,
                    MatrixBlob *LF_C
                    );

  float   ComputeProductSB(       int thread_id,
                    int frame_id,
                    int index_A,
                    int index_C,
                    int latent_index,
                    Tensor3Blob *Sparse_S,
                    MatrixBlob *LF_A,
                    MatrixBlob *LF_B,
                    MatrixBlob *LF_C
                    );

  float   ComputeProductSBC(      int thread_id,
                    int index_A,
                    int latent_index,
                    Tensor3Blob * S,
                    MatrixBlob * B,
                    MatrixBlob * C
                    );


  // Used in ComputeScore
  float   ComputeProductPsiPsiS(  int thread_id,
                      int frame_id,
                      int index_A,
                      Tensor3Blob *Sparse_S,
                      std::vector<int> *indices_B,
                      std::vector<int> *indices_C);

  float   ComputeProductPsiPsiABC(   int thread_id,
                      int frame_id,
                      int index_A,
                      MatrixBlob *LF_A,
                      MatrixBlob *LF_B,
                      MatrixBlob *SLF_B,
                      MatrixBlob *LF_C,
                      std::vector<int> *indices_B,
                      std::vector<int> *indices_C);

  void  DebuggingTestProbe(int thread_id, int index_A, int frame_id, int ground_truth_label, int index);



  virtual float   ComputeScore(int thread_id, int index_A, int frame_id);
  virtual int   ComputeConditionalScore(int thread_id, int index_A, int frame_id, int ground_truth_label);

  void  ComputeWtranspose();

  // void  ComputeWeightW_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  // void  ProcessWeightW_Updates(int thread_id);

  void  ComputeLF_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ProcessLF_A_Updates(int thread_id, int index_A);

  void  ComputeLF_B_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ComputeLF_B_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel);
  void  ProcessLF_B_Updates(int thread_id, int index_A);

  void  ComputeSLF_B_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ProcessSLF_B_Updates(int thread_id, int index_A);

  void  ComputeLF_C_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ComputeLF_C_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel);
  void  ProcessLF_C_Updates(int thread_id, int index_A);

  void  ComputeSparse_S_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ComputeSparse_S_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel);
  void  ProcessSparse_S_Updates(int thread_id, int index_A);

  void  Sparse_S_RegularizationUpdate(int thread_id);

  void  ComputeBias_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ProcessBias_A_Updates(int thread_id, int index_A);

  void  Load_SnapshotWeights(string weights_snapshot_file, string momentum_snapshot_file);
  void  LoadSeedMatricesAndInitializeSparseMatS();

  void  Store_Snapshot(int cross_val_run, int epoch, string fp_snapshot);

  void  ApplyNestorovMomentum(int batch);
};

#endif



