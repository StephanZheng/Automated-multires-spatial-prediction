// Copyright 2014 Stephan Zheng

#include "core/ThreeMatrixFactorTrainer.h"

using namespace std;
using namespace rapidjson;
using std::min;
using std::chrono::high_resolution_clock;

void ThreeMatrixFactorTrainer::Load_SnapshotWeights(string weights_snapshot_file, string momentum_snapshot_file) {
  PrintFancy(Settings_->session_start_time, "Loading weights / biases snapshot " + weights_snapshot_file);
  PrintFancy(Settings_->session_start_time, "Loading momentum     snapshot " + momentum_snapshot_file);

  Settings  *Settings_  = this->Settings_;
  Tensor3Blob *WeightW  = &this->WeightW;
  MatrixBlob  *LF_A     = &this->LF_A;
  MatrixBlob  *LF_B     = &this->LF_B;
  MatrixBlob  *SLF_B    = &this->SLF_B;
  MatrixBlob  *LF_C     = &this->LF_C;
  Tensor3Blob *Sparse_S = &this->Sparse_S;
  VectorBlob  *Bias_A   = &this->Bias_A;

  // Verify dimensions
  assert(Settings_->NumberOfLatentDimensions == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_A->rows  == Settings_->Dimension_A);
  assert(LF_A->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_B->rows  == Settings_->StageThree_Dimension_B);
  assert(LF_B->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_C->rows  == Settings_->StageThree_Dimension_C);
  assert(LF_C->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(Bias_A->columns == Settings_->Dimension_A);

  std::vector<float> *NestorovMomentumLambda = &this->NestorovMomentumLambda;

  string  fn;
  FILE  *fp;

  fn = weights_snapshot_file;
  fp = fopen(fn.c_str() , "rb");

  if (fp == NULL) {
    fputs("Weights file not found!", stderr);
    exit(1);
  }

  int snapshot_file_size = LF_A->data.size() + LF_B->data.size() + LF_C->data.size() + Bias_A->data.size();

  LoadSnapshotWeight(LF_A, weights_snapshot_file, snapshot_file_size, 0, 0, LF_A->rows, 0, LF_A->columns );
  LoadSnapshotWeight(LF_B, weights_snapshot_file, snapshot_file_size, LF_A->data.size(), 0, LF_B->rows, 0, LF_B->columns );
  LoadSnapshotWeight(LF_C, weights_snapshot_file, snapshot_file_size, LF_A->data.size() + LF_B->data.size(), 0, LF_C->rows, 0, LF_C->columns );
  if (Settings_->StageThree_UseSparseMatS == 1) {
    LoadSnapshotWeight(Sparse_S, weights_snapshot_file, snapshot_file_size, LF_A->data.size() + LF_B->data.size() + LF_C->data.size(), 0, Sparse_S->slices, 0, Sparse_S->rows, 0, Sparse_S->columns );
    LoadSnapshotWeight(Bias_A, weights_snapshot_file, snapshot_file_size, LF_A->data.size() + LF_B->data.size() + LF_C->data.size() + Sparse_S->data.size(), 0, Bias_A->columns );
  }
  else {
    LoadSnapshotWeight(Bias_A, weights_snapshot_file, snapshot_file_size, LF_A->data.size() + LF_B->data.size() + LF_C->data.size(), 0, Bias_A->columns );
  }

  fn = momentum_snapshot_file;

  // If we didn't specify a momentum snapshot, we just skip it
  if (fn.length() == 0) {
    PrintFancy(Settings_->session_start_time, "No momentum snapshot specified, skipping loading momentum snapshot");
    return;
  }

  fp = fopen(fn.c_str() , "rb");
  float   *float_ = new float;
  while (fread(float_, sizeof(float), 1, fp)) NestorovMomentumLambda->push_back(*float_);
  fclose(fp);

  delete float_;
  PrintFancy(Settings_->session_start_time, "Resuming training from loaded snapshot");
}
void ThreeMatrixFactorTrainer::Store_Snapshot(int cross_val_run, int epoch, string fp_snapshot) {

  Settings *Settings_         = this->Settings_;
  CurrentStateBlob *CurrentStateBlob_ = this->CurrentStateBlob_;

  // Find index of last recorded loss
  int current_iteration = cross_val_run * Settings_->StageThree_NumberOfCrossValidationRuns + epoch;
  int n_losses_were_recorded = static_cast<int>(floor(static_cast<float>(current_iteration) / Settings_->StageThree_ComputeLossAfterEveryNthEpoch));

  int last_recorded_loss_index = this->Results_LossTrain.size() - 1;

  // cout << "DEBUG:" << this->Results_LossTrain.size() << endl;

  Tensor3Blob *tt;
  MatrixBlob *mm;
  VectorBlob *vvv;
  std::vector<int> *vv;
  std::vector<float> *ff;

  ofstream myfile;
  myfile.open(fp_snapshot + "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_" + "__PYID___" + Settings_->PYTHON_LOGGING_ID +"_settings.json");

  myfile << "Settings_->PYTHON_LOGGING_ID)        : " << Settings_->PYTHON_LOGGING_ID << ",\n";
  myfile << "StageThree_StartingLearningRate        : " << Settings_->StageThree_StartingLearningRate << ",\n";
  myfile << "StageThree_MiniBatchSize           : " << Settings_->StageThree_MiniBatchSize << ",\n";
  myfile << "StageThree_Regularization_LF_A         : " << Settings_->StageThree_Regularization_LF_A << ",\n";
  myfile << "StageThree_Regularization_LF_B_Level1    : " << Settings_->StageThree_Regularization_LF_B_Level1 << ",\n";
  myfile << "StageThree_Regularization_LF_B_Level2    : " << Settings_->StageThree_Regularization_LF_B_Level2 << ",\n";
  myfile << "StageThree_Regularization_LF_C_Level1    : " << Settings_->StageThree_Regularization_LF_C_Level1 << ",\n";
  myfile << "StageThree_Regularization_LF_C_Level2    : " << Settings_->StageThree_Regularization_LF_C_Level2 << ",\n";
  myfile << "StageThree_Regularization_Sparse_S_Level1  : " << Settings_->StageThree_Regularization_Sparse_S_Level1 << ",\n";
  myfile << "StageThree_Regularization_Sparse_S_Level2  : " << Settings_->StageThree_Regularization_Sparse_S_Level2 << ",\n";
  myfile << "StageThree_UpdateSparseMatS_EveryNthMinibatch: " << Settings_->StageThree_UpdateSparseMatS_EveryNthMinibatch << ",\n";
  myfile << "StageThree_ApplyMomentumEveryNthMinibatch  : " << Settings_->StageThree_ApplyMomentumEveryNthMinibatch << ",\n";
  myfile << "StageThree_ResetMomentumEveryNthEpoch    : " << Settings_->StageThree_ResetMomentumEveryNthEpoch << ",\n";
  myfile << "StageThree_Initialization_Sparse_S_range   : " << Settings_->StageThree_Initialization_Sparse_S_range << ",\n";
  myfile << "StageThree_TrainOnLatentFactorsEveryBatch  : " << Settings_->StageThree_TrainOnLatentFactorsEveryBatch << ",\n";
  myfile << "NumberOfLatentDimensions           : " << Settings_->NumberOfLatentDimensions << ",\n";
  myfile << "StageThree_RegularizeS_EveryBatch    : " << Settings_->StageThree_RegularizeS_EveryBatch << ",\n";

  myfile.close();


  time_t now = time(0);
  char timestamp[80];
  strftime(timestamp, sizeof(timestamp), "%d-%b-%Y_%H-%M-%S", localtime(&now));
  string  fn;
  FILE  *fp;

  // We write the current settings ONCE per program run
  if (CurrentStateBlob_->curr_snapshot_id == 0) {
    fn = fp_snapshot + \
          "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_" + \
          "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" +  \
          "_LR_" + to_string(Settings_->StageThree_StartingLearningRate) + "_" + \
          "_LATENTDIM_" + to_string(Settings_->NumberOfLatentDimensions) + "_" + \
          "_REGS_" + to_string(Settings_->StageThree_Regularization_Sparse_S_Level1) + \
          ".override_settings";
    fp = fopen( fn.c_str() , "wb");
    const char c = '!';
    fwrite(&c, sizeof(char), 1, fp);
    fclose(fp);

    // Shuffled indices -- how do we loop throuth the groundtruthlabels?
    fn = fp_snapshot + \
        "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_"+ \
        "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
        to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
        "tv_indices" + "_" + \
        to_string(cross_val_run) + "_" + \
        to_string(epoch) + "_" + \
        ".bin";

    fp = fopen( fn.c_str() , "wb");

    vv = &this->CurrentTrainIndices_Strong;
    for (int i = 0; i < vv->size(); ++i) fwrite(&vv->at(i), sizeof(int), 1, fp);
    vv = &this->CurrentValidIndices_Strong;
    for (int i = 0; i < vv->size(); ++i) fwrite(&vv->at(i), sizeof(int), 1, fp);
    vv = &this->CurrentTrainIndices_Weak;
    for (int i = 0; i < vv->size(); ++i) fwrite(&vv->at(i), sizeof(int), 1, fp);
    vv = &this->CurrentValidIndices_Weak;
    for (int i = 0; i < vv->size(); ++i) fwrite(&vv->at(i), sizeof(int), 1, fp);

    fclose(fp);

  }

  // Write current train indices + computed losses
  fn = fp_snapshot + \
        "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_"+ \
        "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
        to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
        "lossrecord" + "_" + \
        to_string(cross_val_run) + "_" + \
        to_string(epoch) + "_" +\
        to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
        to_string(this->Results_LossValid[last_recorded_loss_index]) + \
        ".bin";

  fp = fopen( fn.c_str() , "wb");

  ff = &this->Results_LossTrain;
  for (int i = 0; i < ff->size(); ++i) fwrite(&ff->at(i), sizeof(float), 1, fp);
  ff = &this->Results_LossValid;
  for (int i = 0; i < ff->size(); ++i) fwrite(&ff->at(i), sizeof(float), 1, fp);
  // ff = &this->Results_LossTest;
  // for (int i = 0; i < ff->size(); ++i) fwrite(&ff->at(i), sizeof(float), 1, fp);

  fclose(fp);



  // Write learned weights + other hyperparameters
  fn =  fp_snapshot + \
      "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_"+ \
      "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
      to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
      "wghts_bias" + "_" + \
      to_string(cross_val_run) + "_" + \
      to_string(epoch) + "_" + \
      to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
      to_string(this->Results_LossValid[last_recorded_loss_index]) +\
      ".bin";

  fp = fopen( fn.c_str(), "wb");

  mm = &this->LF_A;
  for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);
  // mm = &this->LF_A_DiffSq_Cache;
  // for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);

  mm = &this->LF_B;
  for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);
  // mm = &this->LF_B_DiffSq_Cache;
  // for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);

  if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
    mm = &this->SLF_B;
    for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);
  }

  mm = &this->LF_C;
  for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);
  // mm = &this->LF_C_DiffSq_Cache;
  // for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);

  if (Settings_->StageThree_UseSparseMatS == 1){
    tt = &this->Sparse_S;
    for (int i = 0; i < tt->data.size(); ++i) fwrite(&tt->data[i], sizeof(float), 1, fp);
    // tt = &this->Sparse_S_DiffSq_Cache;
    // for (int i = 0; i < tt->data.size(); ++i) fwrite(&tt->data[i], sizeof(float), 1, fp);
  }

  vvv = &this->Bias_A;
  for (int i = 0; i < vvv->data.size(); ++i) fwrite(&vvv->data[i], sizeof(float), 1, fp);
  // vvv = &this->Bias_A_DiffSq_Cache;
  // for (int i = 0; i < vvv->data.size(); ++i) fwrite(&vvv->data[i], sizeof(float), 1, fp);

  fclose(fp);

  // Write ExponentialScores to file
  // fn =  fp_snapshot + \
  //     "/rank3/stage3/" + Settings_->DATASET_NAME + "_r3s3_"+ \
  //     "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
  //     to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
  //     "compscores" + "_" + \
  //     to_string(cross_val_run) + "_" + \
  //     to_string(epoch) + "_" + \
  //     to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
  //     to_string(this->Results_LossValid[last_recorded_loss_index]) +\
  //     ".bin";

  // fp = fopen(fn.c_str(), "wb");

  // mm = &this->ConditionalScore;
  // for (int i = 0; i < mm->data.size(); ++i) fwrite(&mm->data[i], sizeof(float), 1, fp);
  // fclose(fp);

  CurrentStateBlob_->curr_snapshot_id++;
}
// Load biases from completed Stage 1 run
// Load seed matrices from externally performed tensor factorization
void ThreeMatrixFactorTrainer::LoadSeedMatricesAndInitializeSparseMatS() {
  Settings *Settings_   = this->Settings_;

  PrintFancy(Settings_->session_start_time, "Loading bias         seed " + Settings_->StageThree_SnapshotWeightsBiases);
  PrintFancy(Settings_->session_start_time, "Loading LF ballhandler   A seed " + Settings_->StageThree_LF_A_seed);
  PrintFancy(Settings_->session_start_time, "Loading LF Fballhandler  B seed " + Settings_->StageThree_LF_B_seed);
  PrintFancy(Settings_->session_start_time, "Loading LF defender    C seed " + Settings_->StageThree_LF_C_seed);

  if (Settings_->StageThree_UseSparseConcatenation == 1){
    PrintFancy(Settings_->session_start_time, "Loading LF ballhandler  SA seed " + Settings_->StageThree_LF_SA_seed);
    PrintFancy(Settings_->session_start_time, "Loading LF Fballhandler SB seed " + Settings_->StageThree_LF_SB_seed);
    PrintFancy(Settings_->session_start_time, "Loading LF defender   SC seed " + Settings_->StageThree_LF_SC_seed);
  }

  VectorBlob *Bias_A      = &this->Bias_A;
  MatrixBlob *LF_A        = &this->LF_A;
  MatrixBlob *LF_B        = &this->LF_B;
  MatrixBlob *SLF_B       = &this->SLF_B;
  MatrixBlob *LF_C        = &this->LF_C;
  Tensor3Blob *SparseMatS = &this->Sparse_S;

  std::random_device rd;
  std::mt19937 e2(rd());

  // Init RNG for sparse latent factors
  std::uniform_real_distribution<> dist_A(-Settings_->StageThree_A_SubDimension_2_InitRange, Settings_->StageThree_A_SubDimension_2_InitRange);
  std::uniform_real_distribution<> dist_B(-Settings_->StageThree_B_SubDimension_2_InitRange, Settings_->StageThree_B_SubDimension_2_InitRange);
  std::uniform_real_distribution<> dist_C(-Settings_->StageThree_C_SubDimension_2_InitRange, Settings_->StageThree_C_SubDimension_2_InitRange);

  // std::uniform_real_distribution<> dist_A(0.0, Settings_->StageThree_A_SubDimension_2_InitRange);
  // std::uniform_real_distribution<> dist_B(0.0, Settings_->StageThree_B_SubDimension_2_InitRange);
  // std::uniform_real_distribution<> dist_C(0.0, Settings_->StageThree_C_SubDimension_2_InitRange);

  // Assertions on the dataholders in our model
  assert(Settings_->NumberOfLatentDimensions == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_A->rows  == Settings_->Dimension_A);
  assert(LF_A->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_B->rows  == Settings_->StageThree_Dimension_B);
  assert(LF_B->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  assert(LF_C->rows  == Settings_->StageThree_Dimension_C);
  assert(LF_C->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);


  string  fn;
  FILE  *fp;
  float   *float_ = new float;
  size_t  result;
  int   n_floats_in_file;


  int snapshot_file_size = GetExpectedSizeSnapshotFile(Settings_);

  PrintDelimiter(1, 0, 40, '=');
  cout << "Loading bias seed" << endl;
  PrintDelimiter(0, 0, 10, '=');
  cout << "Using dimensions:" << endl;
  cout << "Settings_->Dimension_A:      " << Settings_->Dimension_A << endl;
  cout << "Settings_->StageOne_Dimension_B: " << Settings_->StageOne_Dimension_B << endl;
  cout << "Settings_->StageOne_Dimension_C: " << Settings_->StageOne_Dimension_C << endl;
  PrintDelimiter(0, 0, 10, '=');
  cout << "Total expected floats:       " << snapshot_file_size << endl;
  PrintDelimiter(0, 1, 40, '=');


  fn = Settings_->StageThree_SnapshotWeightsBiases;

  LoadSnapshotWeight(Bias_A, Settings_->StageThree_SnapshotWeightsBiases, snapshot_file_size, snapshot_file_size - Settings_->Dimension_A, 0, Bias_A->columns);

  LoadSnapshotWeight(LF_A, Settings_->StageThree_LF_A_seed, Settings_->Dimension_A * Settings_->StageThree_SubDimension_1 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS, 0, 0, LF_A->rows, 0, Settings_->StageThree_SubDimension_1 );
  LoadSnapshotWeight(LF_B, Settings_->StageThree_LF_B_seed, Settings_->StageThree_Dimension_B * Settings_->StageThree_SubDimension_1 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS, 0, 0, LF_B->rows, 0, Settings_->StageThree_SubDimension_1 );
  LoadSnapshotWeight(LF_C, Settings_->StageThree_LF_C_seed, Settings_->StageThree_Dimension_C * Settings_->StageThree_SubDimension_1 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS, 0, 0, LF_C->rows, 0, Settings_->StageThree_SubDimension_1 );

  if (Settings_->StageThree_UseSparseConcatenation == 1){
    LoadSnapshotWeight(LF_A, Settings_->StageThree_LF_SA_seed, Settings_->Dimension_A * Settings_->StageThree_SubDimension_2 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS,      0, 0, LF_A->rows, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2 );
    LoadSnapshotWeight(LF_B, Settings_->StageThree_LF_SB_seed, Settings_->StageThree_Dimension_B * Settings_->StageThree_SubDimension_2 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS, 0, 0, LF_B->rows, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2 );
    LoadSnapshotWeight(LF_C, Settings_->StageThree_LF_SC_seed, Settings_->StageThree_Dimension_C * Settings_->StageThree_SubDimension_2 / HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS, 0, 0, LF_C->rows, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2 );
  }

  if (Settings_->StageThree_UseSparseMatS == 1)
  {
    std::uniform_real_distribution<> dist_S(0, Settings_->StageThree_Initialization_Sparse_S_range);
    for (int i = 0; i < SparseMatS->data.size(); ++i) {
      if (Settings_->StageThree_UseSparseMatS == 1) {
        SparseMatS->data[i] = dist_S(e2);
      } else {
        SparseMatS->data[i] = 0.;
      }
      if (i < 2) PrintFancy(Settings_->session_start_time, "[SAMPLE] SparseMatSinit: " + to_string(SparseMatS->data[i]));
    }
  }

  delete float_;
  PrintFancy(Settings_->session_start_time, "Loaded training seeds for W, psi.");
  PrintFancy(Settings_->session_start_time, "Initialized sparse matrix S with range " + to_string(Settings_->StageThree_Initialization_Sparse_S_range));
}
void ThreeMatrixFactorTrainer::InitializeWeightsRandom(float range_W, float range_A, float range_psi, float range_S) {
  Settings *Settings_     = this->Settings_;
  MatrixBlob *LF_A        = &this->LF_A;
  MatrixBlob *LF_B        = &this->LF_B;
  MatrixBlob *LF_C        = &this->LF_C;
  VectorBlob *Bias_A      = &(this->Bias_A);
  Tensor3Blob *SparseMatS = &(this->Sparse_S);

  LF_A->initRandom(range_A);
  LF_B->initRandom(range_A);
  LF_C->initRandom(range_A);
  if (Settings_->StageThree_UseSparseMatS == 1){
    SparseMatS->initRandom(range_S);
  }
  Bias_A->initRandom(range_A);
}
void ThreeMatrixFactorTrainer::StoreStartingWeights() {
  PrintFancy(Settings_->session_start_time, "Storing weight snapshots");
  Settings *Settings_      = this->Settings_;
  MatrixBlob  *LF_A        = &this->LF_A;
  MatrixBlob  *LF_B        = &this->LF_B;
  MatrixBlob  *SLF_B       = &this->SLF_B;
  MatrixBlob  *LF_C        = &this->LF_C;
  Tensor3Blob *Sparse_S      = &this->Sparse_S;
  VectorBlob  *Bias_A      = &this->Bias_A;

  MatrixBlob  *LF_A_Snapshot   = &this->LF_A_Snapshot;
  MatrixBlob  *LF_B_Snapshot  = &this->LF_B_Snapshot;
  MatrixBlob  *SLF_B_Snapshot  = &this->SLF_B_Snapshot;
  MatrixBlob  *LF_C_Snapshot   = &this->LF_C_Snapshot;
  Tensor3Blob *Sparse_S_Snapshot = &this->Sparse_S_Snapshot;
  VectorBlob  *Bias_A_Snapshot   = &this->Bias_A_Snapshot;

  for (int i = 0; i < LF_A->data.size(); ++i) LF_A_Snapshot->data[i] = LF_A->data[i];
  for (int i = 0; i < LF_B->data.size(); ++i) LF_B_Snapshot->data[i] = LF_B->data[i];
  if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
    for (int i = 0; i < SLF_B->data.size(); ++i) SLF_B_Snapshot->data[i] = SLF_B->data[i];
  }
  for (int i = 0; i < LF_C->data.size(); ++i) LF_C_Snapshot->data[i] = LF_C->data[i];
  if (Settings_->StageThree_UseSparseMatS == 1) {
    for (int i = 0; i < Sparse_S->data.size(); ++i) Sparse_S_Snapshot->data[i] = Sparse_S->data[i];
  }
  for (int i = 0; i < Bias_A->data.size(); ++i) Bias_A_Snapshot->data[i] = Bias_A->data[i];
}
void ThreeMatrixFactorTrainer::RestoreStartingWeights() {
  PrintFancy(Settings_->session_start_time, "Restoring weight snapshots");
  Settings *Settings_      = this->Settings_;
  MatrixBlob  *LF_A        = &this->LF_A;
  MatrixBlob  *LF_B        = &this->LF_B;
  MatrixBlob  *SLF_B        = &this->SLF_B;
  MatrixBlob  *LF_C        = &this->LF_C;
  Tensor3Blob *Sparse_S      = &this->Sparse_S;
  VectorBlob  *Bias_A      = &this->Bias_A;

  MatrixBlob  *LF_A_Snapshot   = &this->LF_A_Snapshot;
  MatrixBlob  *LF_B_Snapshot   = &this->LF_B_Snapshot;
  MatrixBlob  *SLF_B_Snapshot   = &this->SLF_B_Snapshot;
  MatrixBlob  *LF_C_Snapshot   = &this->LF_C_Snapshot;
  Tensor3Blob *Sparse_S_Snapshot = &this->Sparse_S_Snapshot;
  VectorBlob  *Bias_A_Snapshot   = &this->Bias_A_Snapshot;

  for (int i = 0; i < LF_A->data.size(); ++i) LF_A->data[i] = LF_A_Snapshot->data[i];
  for (int i = 0; i < LF_B->data.size(); ++i) LF_B->data[i] = LF_B_Snapshot->data[i];
  if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
    for (int i = 0; i < SLF_B->data.size(); ++i) SLF_B->data[i] = SLF_B_Snapshot->data[i];
  }
  for (int i = 0; i < LF_C->data.size(); ++i) LF_C->data[i] = LF_C_Snapshot->data[i];
  if (Settings_->StageThree_UseSparseMatS == 1) {
    for (int i = 0; i < Sparse_S->data.size(); ++i) Sparse_S->data[i] = Sparse_S_Snapshot->data[i];
  }
  for (int i = 0; i < Bias_A->data.size(); ++i) Bias_A->data[i] = Bias_A_Snapshot->data[i];

  // cout << "Restored LF _Task data:" << endl;
  // for (int i = 0; i < LF_A->data.size(); ++i) {
  //   cout << LF_A_Snapshot->data[i] << " " << LF_A->data[i] << endl;
  // }
}
int ThreeMatrixFactorTrainer::TrainStageThree(string fp_snapshot) {
  Settings      *Settings_                     = this->Settings_;
  CurrentStateBlob  *CurrentStateBlob          = this->CurrentStateBlob_;

  MatrixBlob      *LF_A                        = &this->LF_A;
  MatrixBlob      *LF_B                        = &this->LF_B;
  MatrixBlob      *LF_C                        = &this->LF_C;
  Tensor3Blob     *Sparse_S                    = &this->Sparse_S;
  VectorBlob      *Bias_A                      = &this->Bias_A;

  MatrixBlob      *LF_A_Diff                   = &this->LF_A_Diff;
  MatrixBlob      *LF_B_Diff                   = &this->LF_B_Diff;
  MatrixBlob      *SLF_B_Diff                  = &this->SLF_B_Diff;
  MatrixBlob      *LF_C_Diff                   = &this->LF_C_Diff;
  Tensor3Blob     *Sparse_S_Diff               = &this->Sparse_S_Diff;
  VectorBlob      *Bias_A_Diff                 = &this->Bias_A_Diff;

  std::vector<int>  *RandomizedIterator        = &this->RandomizedIterator;
  std::vector<float>  *NestorovMomentumLambda_ = &this->NestorovMomentumLambda;

  int * have_recorded_a_trainloss = new int;
  *have_recorded_a_trainloss      = 0;

  int exit_code                   = 0;
  int spcheck_exit_training       = 0;
  int spcheck_restart_training    = 0;

  CurrentStateBlob->current_stage = 3;




  // Debug mode?
  if (Settings_->EnableDebugPrinter_Level1 == 1) {
    EnableDebugMode();
    spatial_entropy.EnableDebugMode();
  } else {
    DisableDebugMode();
    spatial_entropy.DisableDebugMode();
  }

  // Log files for this session
  string logfile_suffix = "_3factor" + to_string(Settings_->StageThree_Dimension_B) + "_" + to_string(CurrentStateBlob_->session_id);
  // Loss
  string logfile_loss_filename = Settings_->LogFolder + Settings_->LogFile_Loss + logfile_suffix;
  IOController_->OpenNewFile(logfile_loss_filename);

  // Entropy
  string logfile_entropy_filename = Settings_->LogFolder + Settings_->LogFile_CellEntropy + logfile_suffix;
  IOController_->OpenNewFile(logfile_entropy_filename);

  // probs
  string logfile_probs_filename = Settings_->LogFolder + Settings_->LogFile_Probabilities + logfile_suffix;
  IOController_->OpenNewFile(logfile_probs_filename);



  // Set the correct number of threads to use in generateTrainingBatches -- this is a slight hack
  Settings_->CurrentNumberOfThreads = Settings_->StageThree_NumberOfThreads;

  // Initialize the parameters to be learned
  if (Settings_->StageThree_StartFromSeed == 1) {
    this->LoadSeedMatricesAndInitializeSparseMatS();
  } else {
    assert(Settings_->StageThree_StartFromSeed == 0);
    if (Settings_->ResumeTraining == 1) {
      this->Load_SnapshotWeights(Settings_->StageThree_SnapshotWeightsBiases, Settings_->StageThree_SnapshotMomentum);
    } else {
      PrintFancy(Settings_->session_start_time, "Random initialization");
      LF_A->initRandom(Settings_->StageThree_Initialization_LF_A_range);
      LF_B->initRandom(Settings_->StageThree_Initialization_LF_A_range);
      LF_C->initRandom(Settings_->StageThree_Initialization_LF_A_range);
      if (Settings_->StageThree_UseSparseMatS == 1){
        Sparse_S->initRandom(Settings_->StageThree_Initialization_Sparse_S_range);
      }
      Bias_A->initRandom(Settings_->StageThree_Initialization_Bias_A_range);
    }
  }


  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Debug -- Bias_A initialization peek");
  Bias_A->showVectorContents(10, 1);

  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Debug -- LF_A initialization peek");
  LF_A->showMatrixContents(0, Settings_->Dimension_A, 0, std::min(LF_A->columns, 10), 100);
  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Debug -- LF_B subdim 1 initialization peek");
  LF_B->showMatrixContents(0, Settings_->StageThree_Dimension_B, 0, std::min(LF_A->columns, 10), 200);
  PrintFancy(Settings_->session_start_time, "Debug -- LF_B subdim 2 initialization peek");
  LF_B->showMatrixContents(0, Settings_->StageThree_Dimension_B, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + std::min(Settings_->StageThree_SubDimension_2, 10), 200);
  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Debug -- LF_C initialization peek");
  LF_C->showMatrixContents(0, Settings_->StageThree_Dimension_C, 0, std::min(LF_A->columns, 10), 50);
  PrintDelimiter(1, 1, 80, '=');

  // Start worker threads
  std::vector<boost::thread *> threads(Settings_->StageThree_NumberOfThreads);
  for (int thread_id = 0; thread_id < Settings_->StageThree_NumberOfThreads; ++thread_id) {
    threads[thread_id] = new boost::thread(&ThreeMatrixFactorTrainer::ThreadComputer, this, thread_id);
  }

  // Start putting tasks on the queue
  int n_total_labels_train = Settings_->NumberOfWeakLabels_Train + Settings_->NumberOfStrongLabels_Train;

  for (int cross_val_run = Settings_->StageThree_StartFromCrossValRun; cross_val_run < Settings_->StageThree_NumberOfCrossValidationRuns; ++cross_val_run) {

    // Clean our record of losses of previous xvalrun
    *have_recorded_a_trainloss = 0;

    PrintFancy(Settings_->session_start_time, "M | Starting cross-validation run: " + to_string(cross_val_run));
    PrintFancy(Settings_->session_start_time, "M | ================================\n");
    CurrentStateBlob->current_cross_val_run = cross_val_run;

    // We do cross-validation, so every x-validation run, we choose a different 10% (the next 10% in the train+val set) as the validation set
    this->PermuteTrainValIndices();

    // A snapshot of the starting weights, to be restored if the scores blow up
    this->StoreStartingWeights();

    // Reset learning learning_rate
    Settings_->StageThree_CurrentLearningRate              = Settings_->StageThree_StartingLearningRate;
    Settings_->StageThree_Cumulative_sum_of_squared_first_derivatives  = 0.;
    Settings_->StageThree_LossTrainPreviousEpoch             = LOSS_INIT_VALUE;
    Settings_->StageThree_LossValidPreviousEpoch             = LOSS_INIT_VALUE;

    int epoch = Settings_->StageThree_StartFromEpoch;

    while (epoch < Settings_->StageThree_NumberOfEpochs) {

      PrintFancy(Settings_->session_start_time, "M | Starting epoch: " + to_string(epoch));
      high_resolution_clock::time_point start_time_epoch = high_resolution_clock::now();
      CurrentStateBlob->current_epoch = epoch;

      int n_batches_served                         = 0;
      int n_batches                                = static_cast<int>(floor(static_cast<float>(n_total_labels_train) / (Settings_->StageThree_NumberOfThreads * Settings_->StageThree_MiniBatchSize) )) + 1;
      int batch_size                               = Settings_->StageThree_NumberOfThreads;
      int last_batch_size                          = 0;
      int n_threads_commanded_this_batch           = 0;
      int n_datapoints_processed_so_far_this_epoch = 0;
      bh_outofbounds_counter                       = 0;

      // Iterator logic:
      // There are gtruth labels, which we partitioned in train-valid-test
      // We shuffled train+valid at the start of xval-runs, and permute the 10% validation through the data-set

      // We then need to go through the train-part randomly!
      // There are two indices.
      // 1. A global index that keeps track of how many training examples we have seen.
      // 2. An index that chooses which element in the train-set to present next.

      // We also want to go randomly through the chosen training-set -- shuffle the train-indices
      // Shuffle the iterators - we do this once every epoch / each run through the entire data-set.
      mt19937 g(static_cast<uint32_t>(time(0)));
      std::shuffle(RandomizedIterator->begin(), RandomizedIterator->end(), g);
      std::shuffle(RandomizedIterator->begin(), RandomizedIterator->end(), g);

      this->PrintStatus();

      if (epoch % Settings_->StageThree_ResetMomentumEveryNthEpoch == 0) {
        PrintFancy(Settings_->session_start_time, "Resetting Nestorov momentum");
        NestorovMomentumLambda_->resize(0);
      }

      PrintFancy(Settings_->session_start_time, "M | Computing terms for train-set. Looping over all strong+weak train labels.");

      for (int batch = 0; batch < n_batches; batch++) {

        if (n_batches_served % Settings_->StageThree_StatusEveryNBatchesTrain == 0) {
          PrintFancy(Settings_->session_start_time, "M | Batch "+to_string(n_batches_served) );
          // this->DebuggingTestProbe(0, 0, 0, 0, 0);
        }

        if (batch % Settings_->TruncatedLasso_Window_UpdateFrequency == 0 and batch > 0) {
          this->UpdateTruncatedLassoWindow(batch);
        }

        // ====================================================================================
        // ComputeConditionalScore
        // ====================================================================================

        // // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeConditionalScore");
        n_threads_commanded_this_batch = this->generateTrainingBatches(1, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        // If scores blow up, adjust learning rate and restart training
        if (global_reset_training == 1) {

          Settings_->StageThree_CurrentLearningRate   *= SCORE_BLOWUP_ADJUST_LR_BY_FACTOR;
          Settings_->StageThree_StartingLearningRate  *= SCORE_BLOWUP_ADJUST_LR_BY_FACTOR;

          PrintDelimiter(1, 1, 80, '=');
          cout << setprecision(PRINT_FLOAT_PRECISION_LR) << "Score blowup! Adjusting LR: " << Settings_->StageThree_CurrentLearningRate / SCORE_BLOWUP_ADJUST_LR_BY_FACTOR << " --> "  << Settings_->StageThree_CurrentLearningRate << endl;
          PrintDelimiter(1, 1, 80, '=');

          this->RestoreStartingWeights(); // Reset all weights to before the diffs were applied this batch
          break;              // Break out of the loop over the data-set
        }

        // ====================================================================================
        // Compute weight deltas
        // ====================================================================================

        if ((batch - Settings_->TrainFrequencyOffset_A) % Settings_->StageThree_Train_LF_A_EveryBatch == 0 and batch > 0) {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeLF_A_Update");
          n_threads_commanded_this_batch = this->generateTrainingBatches(10, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor A this batch");
        }

        if ((batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0 and batch > 0) {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeLF_B_Update");
          n_threads_commanded_this_batch = this->generateTrainingBatches(14, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor B this batch");
        }

        if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
          if (batch % Settings_->StageThree_Train_SLF_B_EveryBatch == 0 and batch > 0) {
            // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeSLF_B_Update");
            n_threads_commanded_this_batch = this->generateTrainingBatches(18, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
            task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
          } else {
            // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing SPARSE latent factor B this batch");
          }
        }

        if ((batch - Settings_->TrainFrequencyOffset_C) % Settings_->StageThree_Train_LF_C_EveryBatch == 0 and batch > 0) {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeLF_C_Update");
          n_threads_commanded_this_batch = this->generateTrainingBatches(16, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor C this batch");
        }



        if (Settings_->StageThree_UseSparseMatS == 1) {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeSparse_S_Update");
          n_threads_commanded_this_batch = this->generateTrainingBatches(12, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        }

        if ((batch - Settings_->TrainFrequencyOffset_Bias) % Settings_->StageThree_TrainBiasEveryBatch == 0 and Settings_->StageThree_UseBias == 1 and batch > 0) {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeBias_A_Update");
          n_threads_commanded_this_batch = this->generateTrainingBatches(4, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        }


        if (Settings_->EnableDebugPrinter_Level1 == 1 and (batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0){
          cout << setw(10) << "DEBUG -- LF_B before update | batch " << batch << endl;
          LF_B->showMatrixContents(0, Settings_->StageThree_Dimension_B, 0, Settings_->NumberOfLatentDimensions, 1);
        }



        // Check out what LF_B_diff looks like --> are the level 1 features working?
        // cout << setw(10) << "DEBUG -- LF_B_Diff | batch " << batch << endl;
        // if (Settings_->EnableDebugPrinter_Level3 == 1 and (batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0 and batch & 20 == 0 and batch > 0){
        //   if (Settings_->StageThree_TrainSubDimension_1 == 1) {
        //     PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) +  " Peek LF_B_Diff subdim 1");
        //     LF_B_Diff->showMatrixContents(150, 160, 0, Settings_->StageThree_SubDimension_1, 1);
        //     cout << endl;
        //   }
        //   if (Settings_->StageThree_TrainSubDimension_2 == 1) {
        //     PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) +  " Peek LF_B_Diff subdim 2");
        //     LF_B_Diff->showMatrixContents(150, 160, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2, 1);
        //     cout << endl;
        //   }
        // }

        // ====================================================================================
        // ====================================================================================
        // Apply computed deltas to weights
        // ====================================================================================
        // ====================================================================================

        // Apply computed updates to latent factors
        // ====================================================================================

        if ((batch - Settings_->TrainFrequencyOffset_A) % Settings_->StageThree_Train_LF_A_EveryBatch == 0 and batch > 0) {
          if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | apply deltas to latent factor A");
          n_threads_commanded_this_batch = this->generateTrainingBatches(11, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor A this batch");
        }

        if ((batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0 and batch > 0) {
          if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | apply deltas to latent factor B");
          n_threads_commanded_this_batch = this->generateTrainingBatches(15, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor B this batch");
        }

        if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
          if (batch % Settings_->StageThree_Train_SLF_B_EveryBatch == 0 and batch > 0) {
            if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | apply deltas to SPARSE latent factor B");
            n_threads_commanded_this_batch = this->generateTrainingBatches(19, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
            task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

          } else {
            // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing SPARSE latent factor B this batch");
          }
        }

        if ((batch - Settings_->TrainFrequencyOffset_C) % Settings_->StageThree_Train_LF_C_EveryBatch == 0 and batch > 0) {
          if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | apply deltas to latent factor C");
          n_threads_commanded_this_batch = this->generateTrainingBatches(17, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        } else {
          // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | Freezing latent factor C this batch");
        }



        // Apply computed updates to sparse matrix S
        // ====================================================================================
        if (Settings_->StageThree_UseSparseMatS == 1){

          if (batch % Settings_->StageThree_UpdateSparseMatS_EveryNthMinibatch == 0) {
            // PrintFancy(Settings_->session_start_time, to_string(batch) + " | applying computed deltas to sparse S"); // // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0)
            n_threads_commanded_this_batch = this->generateTrainingBatches(13, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
            task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
          }

          if (batch % Settings_->StageThree_RegularizeS_EveryBatch == 0 and batch > 0) {
            this->Sparse_S_RegularizationUpdate(0);
          }

          if (Settings_->StageThree_Sparse_S_ClampToThreshold == 1 and batch % 5 == 0) {
            Sparse_S->ThresholdValues(-1e99, Settings_->StageThree_SparseMatSThreshold, 0.0);
          }

        }

        // Apply computed updates to Bias_A
        // if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ProcessBias_A_Updates");
        if ((batch - Settings_->TrainFrequencyOffset_Bias) % Settings_->StageThree_TrainBiasEveryBatch == 0 and Settings_->StageThree_UseBias == 1 and batch > 0) {
          if (batch % Settings_->StageThree_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | applying computed deltas to bias");
          n_threads_commanded_this_batch = this->generateTrainingBatches(5, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        }






        // ------------------------------------------------------------------------------------------------------------------------------
        // Compute spatial entropy of gradients
        // ------------------------------------------------------------------------------------------------------------------------------
        // Add gradients of minibatch to histogram -- this is paralellized
        n_threads_commanded_this_batch = this->generateTrainingBatches(20, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        // Compute entropy of accumulated gradients so far.
        // TODO(stz): implement streaming version of this: how to deal with renormalization per update?
        spatial_entropy.ComputeEmpiricalDistribution();
        spatial_entropy.ComputeSpatialEntropy();
        spatial_entropy.LogToFile(Settings_->session_start_time, logfile_probs_filename, logfile_entropy_filename);

        if (debug_mode and batch % 100 == 0) {
          spatial_entropy.ShowEntropies();
        }





        // ====================================================================================
        // Apply momentum
        if (batch % Settings_->StageThree_ApplyMomentumEveryNthMinibatch == 0) this->ApplyNestorovMomentum(batch);
        // ====================================================================================

        // Tune sparsity
        // ====================================================================================
        if (((batch - Settings_->TrainFrequencyOffset_A) % Settings_->TuneSparsityEveryBatch == 0 or \
           (batch - Settings_->TrainFrequencyOffset_B) % Settings_->TuneSparsityEveryBatch == 0 or \
           (batch - Settings_->TrainFrequencyOffset_C) % Settings_->TuneSparsityEveryBatch == 0) \
           and batch > 0 and Settings_->StageThree_TuneSparsity == 1 \
           and (    Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3 \
              or  Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) ) {

          // Get relevant sparsity-levels and decision how to continue
          exit_code = this->tuneSparsityAll(batch, LF_A, LF_B, LF_C);
          assert(exit_code == -1 or exit_code == 0 or exit_code == 1 or exit_code == 2 or exit_code == 3);

          if (exit_code == -1) {
            PrintWithDelimiters(Settings_->session_start_time, "[SparsityCheck] Illegal: you did not provide correct settings.");
            spcheck_exit_training = 1;
            break;
          }
          else if (exit_code == 0) { // If tuneSparsityAll returns 0, sparsity level AND running average are below the desired sparsity level --> we quit.
            PrintWithDelimiters(Settings_->session_start_time, "Desired sparsity level reached --> quitting epoch (and training).");
            spcheck_exit_training = 1;
            break;
          }
          else if (exit_code == 1) { // Gradient is not increasing, so continue with current regularization strength
            PrintWithDelimiters(Settings_->session_start_time, "Desired sparsity level NOT reached + sparsity level gradient not increasing --> continuing training.");
          }
          else if (exit_code == 2) { // Gradient is increasing --> lower regularization and continue
            PrintWithDelimiters(Settings_->session_start_time, "Desired sparsity level NOT reached + sparsity level gradient IS increasing --> lower regularization + continuing training.");
          }
          // If tuneSparsityAll returns 3, sparsity level is < BUT running average is > the desired sparsity level --> we lower L1 regularization strength.
          else if (exit_code == 3) {
            PrintWithDelimiters(Settings_->session_start_time, "Sparsity level is < and running average is < the desired sparsity level, but seen too few sparsity checks --> we lower L1 regularization strength AND restart.");
            spcheck_restart_training = 1;
          }

        }

        if (spcheck_restart_training == 1) {
          this->RestoreStartingWeights(); // Reset all weights to before the diffs were applied this epoch
          break;              // Break out of the loop over the data-set
        }

        // Debug
        // ====================================================================================

        if (Settings_->EnableDebugPrinter_Level3 == 1 and (batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0 and (batch - Settings_->TrainFrequencyOffset_B) % 30 == 0 and batch > 0){

          printf("\n batch %i LF_B after update col %i:%i \n", batch, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
          LF_B->showMatrixContents(400, 420, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2, 1);

          printf("\n batch %i LF_B_Diff after update col %i:%i \n", batch, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
          LF_B_Diff->showMatrixContents(400, 420, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2, 1);

        }

        if (Settings_->EnableDebugPrinter_Level3 == 1 and (batch - Settings_->TrainFrequencyOffset_C) % Settings_->StageThree_Train_LF_C_EveryBatch == 0 and (batch - Settings_->TrainFrequencyOffset_C) % 30 == 0 and batch > 0){

          printf("\n batch %i LF_C after update col %i:%i \n", batch, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
          LF_C->showMatrixContents(50, 60, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2, 1);

          printf("\n batch %i LF_C_Diff after update col %i:%i \n", batch, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
          LF_C_Diff->showMatrixContents(50, 60, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2, 1);
        }

        // Clamp weights to be positive
        // ====================================================================================
        // LF_A->ThresholdValues(-1e99, 0.0, 0.0);
        // LF_B->ThresholdValues(-1e99, 0.0, 0.0);
        // LF_C->ThresholdValues(-1e99, 0.0, 0.0);
        // LF_B->showMatrixContents(5, 1);

        // Erase the deltas computed this batch
        // ====================================================================================
        LF_A_Diff->erase();
        LF_B_Diff->erase();
        if (Settings_->StageThree_UseSparseLatent_B_Term == 1) SLF_B_Diff->erase();
        LF_C_Diff->erase();
        if (Settings_->StageThree_UseSparseMatS == 1 and batch % Settings_->StageThree_UpdateSparseMatS_EveryNthMinibatch == 0) {
          Sparse_S_Diff->erase();
        }
        Bias_A_Diff->erase();

        // Note this miscounts at the last batch (n_batches - 1), because that batch might contain a smaller number of datapoints than MiniBatchSize
        // but this is irrelevant there anyway [EXCEPT FOR MISUNDERSTANDINGS DURING DEBUGGING]
        n_datapoints_processed_so_far_this_epoch += n_threads_commanded_this_batch * Settings_->StageThree_MiniBatchSize;

        n_batches_served++;
      }

      PrintTimeElapsedSince(start_time_epoch, "Batch train-time: ");

      // If a blow-up was detected OR sparsity was too fast, weights were reset and we skip the loss computations
      if (global_reset_training == 1 or spcheck_restart_training == 1) {
        // Note: if we reset the training (because of blowup), we DO NOT STORE A SNAPSHOT!
        PrintFancy(Settings_->session_start_time, "Erasing recorded sparsity levels");
        // Reset all recorded sparsity levels
        LF_A->sp_sd1_lvls.init(0);
        LF_A->sp_sd1_nonzeros.init(0);
        LF_A->sp_sd2_lvls.init(0);
        LF_A->sp_sd2_nonzeros.init(0);

        LF_B->sp_sd1_lvls.init(0);
        LF_B->sp_sd1_nonzeros.init(0);
        LF_B->sp_sd2_lvls.init(0);
        LF_B->sp_sd2_nonzeros.init(0);

        LF_C->sp_sd1_lvls.init(0);
        LF_C->sp_sd1_nonzeros.init(0);
        LF_C->sp_sd2_lvls.init(0);
        LF_C->sp_sd2_nonzeros.init(0);

        // Reset the epoch counter (but we stay in the current xval-run)
        epoch          = Settings_->StageThree_StartFromEpoch;
        global_reset_training  = 0;
        spcheck_restart_training = 0;
        continue;
      }

      // Compute losses now
      if (epoch % Settings_->StageThree_ComputeLossAfterEveryNthEpoch == 0) {
        high_resolution_clock::time_point start_time_ComputeValidTestSetScores = high_resolution_clock::now();

        PrintFancy(Settings_->session_start_time, "ComputeConditionalScore Validation");
        // TODO(Stephan) MT - We need to recalculate the ExponentialScores (for train, only ExponentialScores for train-set have been done)!
        int n_total_labels_valid                 = Settings_->NumberOfWeakLabels_Val + Settings_->NumberOfStrongLabels_Val;
        n_batches_served                         = 0;
        n_batches                                = static_cast<int>(floor(static_cast<float>(n_total_labels_valid) / (Settings_->StageThree_NumberOfThreads * Settings_->StageThree_MiniBatchSize) )) + 1;
        batch_size                               = Settings_->StageThree_NumberOfThreads;
        last_batch_size                          = 0;
        n_datapoints_processed_so_far_this_epoch = 0;

        for (int batch = 0; batch < n_batches; batch++) {
          if (n_batches_served % Settings_->StageThree_StatusEveryNBatchesValid == 0) PrintFancy(Settings_->session_start_time, "M | Batch "+to_string(n_batches_served) );

          // ComputeConditionalScore
          n_threads_commanded_this_batch = this->generateValidationBatches(1, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageThree_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

          // Note this miscounts at the last batch (n_batches - 1), because that batch might contain a smaller number of datapoints than MiniBatchSize
          // but this is irrelevant there anyway [EXCEPT FOR MISUNDERSTANDINGS DURING DEBUGGING]
          n_datapoints_processed_so_far_this_epoch += n_threads_commanded_this_batch * Settings_->StageThree_MiniBatchSize;

          n_batches_served++;
        }


        this->ComputeConditionalScoreSingleThreaded(8);

        high_resolution_clock::time_point start_time_loss_compute = high_resolution_clock::now();

        // All ExponentialScores have been computed, so we can compute the loss
        // ====================================================================================
        float loss_train = this->ComputeLoss(6);
        float loss_valid = this->ComputeLoss(7);
        float loss_test  = LOSS_INIT_VALUE; //this->ComputeLoss(8);
        this->ProcessComputedLosses(cross_val_run, epoch, loss_train, loss_valid, loss_test);

        PrintDelimiter(1, 1, 80, '=');
        PrintFancy(Settings_->session_start_time, "Epoch loss results -- xval-run: "+to_string(cross_val_run)+" | epoch "+to_string(epoch));

        cout << ">>>> Train loss: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_train;
        cout << " -- prev: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << Settings_->StageOne_LossTrainPreviousEpoch;
        cout << " -- delta: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_train - Settings_->StageOne_LossTrainPreviousEpoch;
        cout << endl;

        cout << ">>>> Valid loss: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_valid;
        cout << " -- prev: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << Settings_->StageOne_LossValidPreviousEpoch;
        cout << " -- delta: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_valid - Settings_->StageOne_LossValidPreviousEpoch;
        cout << endl;
        PrintTimeElapsedSince(start_time_loss_compute, "Loss compute-time: ");
        PrintDelimiter(1, 1, 80, '=');


        // Also showing sparsity if not quitting training
        // ====================================================================================
        if (spcheck_exit_training != 1){
          PrintFancy(Settings_->session_start_time, "We're not quitting training, so showing you sparsity of this epoch");
          PrintFancy(Settings_->session_start_time, "Epoch sparsity results -- xval-run: "+to_string(cross_val_run)+" | epoch "+to_string(epoch));
          exit_code = this->tuneSparsityAll(-1, LF_A, LF_B, LF_C);
          assert(exit_code == -1 or exit_code == 0 or exit_code == 1 or exit_code == 2 or exit_code == 3);
        }

        // Adjust learning-rate etc based on what loss we've seen this epoch
        // ====================================================================================

        if (this->DecisionUnit(cross_val_run, epoch, loss_train, loss_valid, loss_test, have_recorded_a_trainloss) == 0) break;


        // -------------------------------------------------------------------------------------------------
        // Write losses to log-file
        // -------------------------------------------------------------------------------------------------
        IOController_->WriteToFile(logfile_loss_filename, to_string(GetTimeElapsedSince(Settings_->session_start_time)) + ",");
        IOController_->WriteToFile(logfile_loss_filename, to_string(loss_train) + ",");
        IOController_->WriteToFile(logfile_loss_filename, to_string(loss_valid) + "\n");
        // -------------------------------------------------------------------------------------------------

      }

      // Take a snapshot - store the learned parameters
      // ====================================================================================
      if (epoch % Settings_->StageThree_Take_SnapshotAfterEveryNthEpoch == 0 || epoch == Settings_->StageThree_NumberOfEpochs - 1) {
        PrintFancy(Settings_->session_start_time, "M | Taking snapshot | Exporting learned parameters to file -- " + fp_snapshot + "/[filename]");
        this->Store_Snapshot(cross_val_run, epoch, fp_snapshot);
      }

      if (spcheck_exit_training == 1) {
        PrintWithDelimiters(Settings_->session_start_time, "Desired sparsity level reached -- quitting this xval-run.");
        break;
      }


      // Show Entropy summary.
      spatial_entropy.ShowSummary();
      // ------------------------------------------------------------------------------------------------
      // End of epoch
      // ------------------------------------------------------------------------------------------------
      // Loss did not below threshold, but we hit the max #epochs
      if (epoch == Settings_->StageThree_NumberOfEpochs - 1) PrintFancy(Settings_->session_start_time, "Maximal number of epochs reached, but not all loss thresholds reached... continuing to next xval-run.");

      // Go to the next epoch
      epoch++;
    }

    if (spcheck_exit_training == 1) {
      PrintWithDelimiters(Settings_->session_start_time, "Desired sparsity level reached -- quitting training.");
      break;
    }

  }

  // Send worker threads the KILL signal
  for (int i = 0; i < Settings_->StageThree_NumberOfThreads; ++i) {
    QueueMessage QueueMessageSend_;
    QueueMessageSend_.task_type = 0;
    QueueMessageSend_.index_A   = _dummy_index_A_to_test_mt+i;

    // TODO(Stephan) Change this: getTask() routine should read 1 at a time?
    task_queue_.mutex_.lock();
    for (int i = 0; i < Settings_->StageThree_MiniBatchSize; ++i) {
      task_queue_.taskQueue.push(QueueMessageSend_);
    }
    task_queue_.mutex_.unlock();
    task_queue_.qGoFetch.notify_one();
  }

  for (int thread_id = 0; thread_id < Settings_->StageThree_NumberOfThreads; ++thread_id) {
    if (threads[thread_id]->joinable()) {
      PrintFancy(Settings_->session_start_time, "Waiting for thread " + to_string(thread_id));
      threads[thread_id]->join();
    }
  }
  for (int thread_id = 0; thread_id < Settings_->StageThree_NumberOfThreads; ++thread_id) {
    delete threads[thread_id];
  }

  delete have_recorded_a_trainloss;
  PrintFancy(Settings_->session_start_time, "Workers are all done -- end of Stage 3.");
  return 0;
}
int ThreeMatrixFactorTrainer::DecisionUnit(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test, int * have_recorded_a_trainloss){

  Settings *Settings_ = this->Settings_;

  PrintFancy(Settings_->session_start_time, "Deciding what the next step is -- *have_recorded_a_trainloss = " + to_string(*have_recorded_a_trainloss));

  // abs(loss_train) < settings->StageThree_TrainingConditionalScoreThreshold \
  // || abs(loss_valid) < settings->StageThree_ValidationConditionalScoreThreshold \
  // || abs(loss_test) < settings->StageThree_TestConditionalScoreThreshold

  // If validation loss goes up again -- not implemented here
  // if ( loss_valid <= 0 ) {
  //   // Do stuff here
  //   PrintFancy(Settings_->session_start_time, "M | Stage 3: validation loss delta thresholds reached");
  //   PrintFancy(Settings_->session_start_time, "M | Exporting learned parameters to file -- this->Store_Snapshot");
  //   this->Store_Snapshot(cross_val_run, epoch, fp_snapshot);
  //   break;
  // }

  if (*have_recorded_a_trainloss == 1) {

    if (this->CheckIfFloatIsNan(loss_train, "") || this->CheckIfFloatIsNan(loss_valid, "") || abs(loss_train) > LOSS_BLOWUP_THRESHOLD || abs(loss_valid) > LOSS_BLOWUP_THRESHOLD) {

      // Train / valid loss blew up?
      // ====================================================================================

      PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " blew up!");
      PrintFancy(Settings_->session_start_time, "Adjusting learning-rate + resetting weights ");
      cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageThree_CurrentLearningRate << " --> " << Settings_->StageThree_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;

      PrintFancy(Settings_->session_start_time, "Will try a new epoch");
      Settings_->StageThree_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;
      this->RestoreStartingWeights();
    } else if (loss_valid - Settings_->StageThree_LossValidPreviousEpoch > 0 and abs(loss_valid - Settings_->StageThree_LossValidPreviousEpoch) > Settings_->StageThree_ValidationLossIncreaseThreshold) {

      PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " > " +to_string(Settings_->StageOne_LossValidPreviousEpoch));
      PrintFancy(Settings_->session_start_time, "Validation delta over threshold, quitting this xval-run and going to next xval-run.");

      // PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " > " +to_string(Settings_->StageThree_LossValidPreviousEpoch));
      // PrintFancy(Settings_->session_start_time, "Adjusting learning-rate: ");
      // cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageThree_CurrentLearningRate << " --> " << Settings_->StageThree_CurrentLearningRate * LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY << endl;
      // Settings_->StageThree_CurrentLearningRate *= LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY;

      return 0;

    } else if (loss_valid - Settings_->StageThree_LossValidPreviousEpoch < 0 and abs(loss_valid - Settings_->StageThree_LossValidPreviousEpoch) < Settings_->StageThree_ValidationLossDecreaseThreshold) {

      PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " < " +to_string(Settings_->StageOne_LossValidPreviousEpoch));
      PrintFancy(Settings_->session_start_time, "Validation delta under threshold, quitting this xval-run and going to next xval-run.");

      // PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " > " +to_string(Settings_->StageThree_LossValidPreviousEpoch));
      // PrintFancy(Settings_->session_start_time, "Adjusting learning-rate: ");
      // cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageThree_CurrentLearningRate << " --> " << Settings_->StageThree_CurrentLearningRate * LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY << endl;
      // Settings_->StageThree_CurrentLearningRate *= LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY;

      return 0;

    } else {

      // Validation loss did not go up, what did train-loss do?

      if (loss_train > Settings_->StageThree_LossTrainPreviousEpoch) {
        PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " > " +to_string(Settings_->StageThree_LossTrainPreviousEpoch));
        PrintFancy(Settings_->session_start_time, "Decreasing learning-rate: ");
        PrintDelimiter(1, 1, 80, '=');
        cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageThree_CurrentLearningRate << " --> " << Settings_->StageThree_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;
        PrintDelimiter(1, 1, 80, '=');
        Settings_->StageThree_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;
      } else {
        PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " <= " +to_string(Settings_->StageThree_LossTrainPreviousEpoch));
        PrintFancy(Settings_->session_start_time, "Increasing learning-rate: ");
        PrintDelimiter(1, 1, 80, '=');
        cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageThree_CurrentLearningRate << " --> " << Settings_->StageThree_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_DOWN_CHANGE_RATE_BY << endl;
        PrintDelimiter(1, 1, 80, '=');
        Settings_->StageThree_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_DOWN_CHANGE_RATE_BY;
        // Store the better (= lower) training loss and go to next epoch
        Settings_->StageThree_LossTrainPreviousEpoch = loss_train;
        Settings_->StageThree_LossValidPreviousEpoch = loss_valid;

      }
    }
  } else if (*have_recorded_a_trainloss == 0) {

    if (this->CheckIfFloatIsNan(loss_train, "") || this->CheckIfFloatIsNan(loss_valid, "") || abs(loss_train) > LOSS_BLOWUP_THRESHOLD || abs(loss_valid) > LOSS_BLOWUP_THRESHOLD) {

      // Train / valid loss blew up?
      // ====================================================================================

      PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " blew up!");
      PrintFancy(Settings_->session_start_time, "Adjusting learning-rate + resetting weights ");
      cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;

      PrintFancy(Settings_->session_start_time, "Will try again with lower LR");
      Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;

    } else {
      PrintFancy(Settings_->session_start_time, "No previous train / valid loss recorded, and losses did not blow up. Storing and going to next epoch.");
      Settings_->StageThree_LossTrainPreviousEpoch = loss_train;
      Settings_->StageThree_LossValidPreviousEpoch = loss_valid;
      *have_recorded_a_trainloss = 1;

    }

  }

  return 1;
}
void ThreeMatrixFactorTrainer::ThreadComputer(int thread_id) {
  Settings *Settings_ = this->Settings_;
  int MiniBatchSize   = Settings_->StageThree_MiniBatchSize;

  while (1) {
    // We put a pointer to the QueueMessage minibatch array on the queue, which we receive here.
    QueueMessage *tasks = new QueueMessage[MiniBatchSize];
    task_queue_.getTaskMessage(thread_id, tasks, MiniBatchSize);

    // PrintFancy(Settings_->session_start_time, to_string(thread_id)+" fetched tasks, off to work " + to_string(tasks[0].task_type));

    // Now loop over the entries in QueueMessage
    for (int entry = 0; entry < MiniBatchSize; ++entry) {
      // Fetch the task from the minibatch
      QueueMessage task = tasks[entry];

      // Last batch: if number of datapoints is less than minibatchsize, break this loop
      if (task.index_A == -1) break;

      if (thread_id == 1) {
        // PrintFancy(Settings_->session_start_time, "BallHandler_id: " + to_string(task.index_A));
        // PrintFancy(Settings_->session_start_time, "Im_reg_id: " + to_string(task.frame_id));
        // PrintFancy(Settings_->session_start_time, "Freq   : " + to_string(task.ground_truth_label));
      }

      // if (thread_id < 2) PrintFancy(Settings_->session_start_time, to_string(thread_id) + " | QueueMessage " +to_string(entry) + " | index_A     " + to_string(task.index_A));
      // if (thread_id < 2) PrintFancy(Settings_->session_start_time, to_string(thread_id) + " | QueueMessage " +to_string(entry) + " | frame_id " + to_string(task.frame_id));

      if (task.task_type == 0) {  // If we receive 0 (KILL signal) exit the function
        // PrintFancy(Settings_->session_start_time, "Thread: " +to_string(thread_id) + " | exit command seen: " + to_string(task.index_A) + " | Victim #" +to_string(n_victims));
        // n_victims++;
        return;
      }
      if (task.task_type == 1) {
        // PrintFancy(Settings_->session_start_time, "ComputeConditionalScore");
        // PrintFancy(Settings_->session_start_time, "Computing ExponentialScore = multiplicity * (score - gtruthlabel) for (A, J)
        this->ComputeConditionalScore(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      // if (task.task_type == 2) {
      //   // PrintFancy(Settings_->session_start_time, "ComputeWeightW_Update");
      //   this->ComputeWeightW_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      // }
      // if (task.task_type == 3) {
      //   // PrintFancy(Settings_->session_start_time, "ProcessWeightW_Updates");
      //   this->ProcessWeightW_Updates(thread_id);
      // }
      if (task.task_type == 4) {
        // PrintFancy(Settings_->session_start_time, "ComputeBias_A_Update");
        this->ComputeBias_A_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 5) {
        // PrintFancy(Settings_->session_start_time, "ProcessBias_A_Updates");
        this->ProcessBias_A_Updates(thread_id, task.index_A);
      }

      if (task.task_type == 10) {
        this->ComputeLF_A_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 11) {
        this->ProcessLF_A_Updates(thread_id, task.index_A);
      }
      if (task.task_type == 12) {
        this->ComputeSparse_S_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 13) {
        this->ProcessSparse_S_Updates(thread_id, task.index_A);
      }

      if (task.task_type == 14) {
        this->ComputeLF_B_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 15) {
        this->ProcessLF_B_Updates(thread_id, task.index_A);
      }

      if (task.task_type == 16) {
        this->ComputeLF_C_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 17) {
        this->ProcessLF_C_Updates(thread_id, task.index_A);
      }

      if (task.task_type == 18) {
        this->ComputeSLF_B_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 19) {
        this->ProcessSLF_B_Updates(thread_id, task.index_A);
      }

      if (task.task_type == 20) {
        this->ComputeSpatialEntropy(thread_id, task.index_A, task.frame_id, task.ground_truth_label,
          Settings_->StageThree_Dimension_B,
          Settings_->StageThree_Dimension_C,
          1.0,
          Settings_->StageThree_MiniBatchSize);
      }

      // 6 = train-loss
      // 7 = valid-loss
      // 8 = test -loss
      if (task.task_type == 6 || task.task_type == 7 || task.task_type == 8) {
        // PrintFancy(Settings_->session_start_time, "ComputeLossPartial");
        // this->ComputeLossPartial(thread_id, task.labels_id_start, task.labels_id_end, task.index_A_start, task.index_A_end, task.task_type)
      }

    }

    // Worker only gets here if he didn't see a task completion (type 101)
    // Worker is done with this task, so notify the master

    boost::mutex::scoped_lock lock(task_queue_.mutex_complete_);
    task_queue_.taskCompletionQueue.push(thread_id);
    task_queue_.qComplete.notify_one();

    // We don't need the QueueMessage minibatch any more after this, so delete it
    delete[] tasks;
  }
  return;
}
void ThreeMatrixFactorTrainer::DebuggingTestProbe(int thread_id, int index_A, int frame_id, int ground_truth_label, int index) {
  // Settings     *Settings_   = this->Settings_;
  // MatrixBlob *WeightW = &(this->WeightW);
  // MatrixBlob *OccupancyFeat = this->Blob_B_;
  // VectorBlob *Bias_A    = &(this->Bias_A);

  // PrintFancy(Settings_->session_start_time, "Running DebuggingTestProbe");
  // for (int index = 0; index < 8192; ++index) {
  //   if (index % 500 == 0) {
  //     cout << "What's in U? " << index_A << " " << index << " " << WeightW->data[WeightW->SerialIndex(index_A, index)] << endl;
  //   }
  //   if (abs(WeightW->data[WeightW->SerialIndex(index_A, index)]) > 0.001) {
  //     cout << ">>> Big value spotted: " << index_A << " " << index << " " << WeightW->data[WeightW->SerialIndex(index_A, index)] << endl;
  //   }
  // }
}
void ThreeMatrixFactorTrainer::ApplyNestorovMomentum(int batch) {
  Settings      *Settings_              = this->Settings_;
  std::vector<float>  *NestorovMomentumLambda_      = &this->NestorovMomentumLambda;

  std::vector<float>  *NestorovMomentumPreviousLF_A_    = &this->NestorovMomentumPreviousLF_A;
  std::vector<float>  *NestorovMomentumPreviousLF_B_    = &this->NestorovMomentumPreviousLF_B;
  std::vector<float>  *NestorovMomentumPreviousSLF_B_    = &this->NestorovMomentumPreviousSLF_B;
  std::vector<float>  *NestorovMomentumPreviousLF_C_    = &this->NestorovMomentumPreviousLF_C;
  // std::vector<float>  *NestorovMomentumPreviousSparse_S_ = &this->NestorovMomentumPreviousSparse_S;
  std::vector<float>  *NestorovMomentumPreviousBias_A_    = &this->NestorovMomentumPreviousBias_A;

  MatrixBlob *LF_A            = &(this->LF_A);
  MatrixBlob *LF_A_diff         = &(this->LF_A_Diff);
  MatrixBlob *LF_A_DiffSq_Cache     = &(this->LF_A_DiffSq_Cache);

  MatrixBlob *LF_B            = &this->LF_B;
  MatrixBlob *LF_B_Diff         = &this->LF_B_Diff;
  MatrixBlob *LF_B_DiffSq_Cache     = &this->LF_B_DiffSq_Cache;

  MatrixBlob *SLF_B           = &this->SLF_B;
  MatrixBlob *SLF_B_Diff        = &this->SLF_B_Diff;
  MatrixBlob *SLF_B_DiffSq_Cache    = &this->SLF_B_DiffSq_Cache;

  MatrixBlob *LF_C            = &this->LF_C;
  MatrixBlob *LF_C_Diff         = &this->LF_C_Diff;
  MatrixBlob *LF_C_DiffSq_Cache     = &this->LF_C_DiffSq_Cache;

  // Tensor3Blob *Sparse_S        = &(this->Sparse_S);
  // Tensor3Blob *Sparse_S_diff     = &(this->Sparse_S_Diff);
  // Tensor3Blob *Sparse_S_DiffSq_Cache = &(this->Sparse_S_DiffSq_Cache);

  VectorBlob *Bias_A          = &this->Bias_A;
  VectorBlob *Bias_A_Diff         = &this->Bias_A_Diff;
  VectorBlob *Bias_A_DiffSq_Cache     = &this->Bias_A_DiffSq_Cache;

  // PrintFancy(Settings_->session_start_time, "Applying momentum");
  int n_dimension_A           = Settings_->Dimension_A;
  int n_dimension_B           = Settings_->StageThree_Dimension_B;
  int n_dimension_C           = Settings_->StageThree_Dimension_C;
  int n_dimension_latent        = Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2;

  float lambda              = 0.;
  float lambda_new            = 0.;
  float gamma               = 0.;

  // For conventions on old and new lambdas / gammas, see
  // https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
  // PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) + " | doing momentum updates");
  int   n_momentum_updates_seen   = NestorovMomentumLambda_->size();

  if (n_momentum_updates_seen == 0) {
    PrintFancy(Settings_->session_start_time, "No previous momentum updates, initializing now.");
    NestorovMomentumLambda_->push_back(1.);

    if (Settings_->StageThree_UseLatentTerm == 1) {
      // for (int i = 0; i < n_dimension_B * n_dimension_C * n_dimension_latent; ++i) {
      //   NestorovMomentumPreviousWeight_->push_back(WeightW->data[i]);  // Store the first weight snapshot for the momentum updates
      // }
      for (int i = 0; i < LF_A->data.size(); ++i) {
        NestorovMomentumPreviousLF_A_->push_back(LF_A->data[i]);
      }
      for (int i = 0; i < LF_B->data.size(); ++i) {
        NestorovMomentumPreviousLF_B_->push_back(LF_B->data[i]);
      }
      if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
        for (int i = 0; i < LF_B->data.size(); ++i) {
          NestorovMomentumPreviousSLF_B_->push_back(SLF_B->data[i]);
        }
      }
      for (int i = 0; i < LF_C->data.size(); ++i) {
        NestorovMomentumPreviousLF_C_->push_back(LF_C->data[i]);
      }
    }
    // if (Settings_->StageThree_UseSparseMatS == 1) {
    //   for (int i = 0; i < n_dimension_B * n_dimension_C * n_dimension_A; ++i) {
    //     NestorovMomentumPreviousSparse_S_->push_back(Sparse_S->data[i]);  // Store the first weight snapshot for the momentum updates
    //   }
    // }
    if (Settings_->StageThree_UseBias == 1) {
      for (int i = 0; i < n_dimension_A; ++i) {
        NestorovMomentumPreviousBias_A_->push_back(Bias_A->data[i]);  // Store the first weight snapshot for the momentum updates
      }
    }

  } else {  // Apply the momentum update

    // Momentum factor
    lambda   = NestorovMomentumLambda_->at(n_momentum_updates_seen - 1);
    lambda_new = 0.5 * (1 + sqrt(1 + 4 * pow(lambda, 2) ) );

    NestorovMomentumLambda_->push_back(lambda_new);

    // Both W and Wdiff are indexed row-first, so it's safe to just do a serialized for-loop
    if (Settings_->StageThree_UseLatentTerm == 1) {
      // PrintFancy(Settings_->session_start_time, "Applying momentum to weight W and latent factor Psi");
      // this->BoostParameter(WeightW, WeightU_gradient, WeightW_DiffSq_Cache, NestorovMomentumPreviousWeight_, n_dimension_B * n_dimension_C * n_dimension_latent);
      this->BoostParameter(batch, LF_A, LF_A_diff, LF_B_DiffSq_Cache, NestorovMomentumPreviousLF_A_, LF_A->data.size());
      this->BoostParameter(batch, LF_B, LF_B_Diff, LF_B_DiffSq_Cache, NestorovMomentumPreviousLF_B_, LF_B->data.size());

      if (Settings_->StageThree_UseSparseLatent_B_Term == 1) this->BoostParameter(batch, SLF_B, SLF_B_Diff, SLF_B_DiffSq_Cache, NestorovMomentumPreviousSLF_B_, SLF_B->data.size());

      this->BoostParameter(batch, LF_C, LF_C_Diff, LF_C_DiffSq_Cache, NestorovMomentumPreviousLF_C_, LF_C->data.size());
    }
    // if (Settings_->StageThree_UseSparseMatS == 1) {
    //   // PrintFancy(Settings_->session_start_time, "Applying momentum to sparse matrix S");
    //   this->BoostParameter(Sparse_S, Sparse_S_diff, Sparse_S_DiffSq_Cache, NestorovMomentumPreviousSparse_S_, Sparse_S->data.size());
    // }
    if (Settings_->StageThree_UseBias == 1) {
      // PrintFancy(Settings_->session_start_time, "Applying momentum to bias B");
      this->BoostParameter(batch, Bias_A, Bias_A_Diff, Bias_A_DiffSq_Cache, NestorovMomentumPreviousBias_A_, Bias_A->data.size());
    }
  }
}

// Compute for update C
inline float ThreeMatrixFactorTrainer::ComputeProductPsiPsiAB(int thread_id, int frame_id, int index_A, int latent_index, MatrixBlob *LF_A, MatrixBlob *LF_B, std::vector<int> *indices_B) {

  float sum = 0.;
  Settings *Settings_ = this->Settings_;

  // Note that we skip the sentinel value at the end!
  for (int index_B = 0; index_B < indices_B->size() - 1; ++index_B)
  {
    SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S3
    if (indices_B->at(index_B) < 0) continue;

    sum +=  LF_A->at(index_A, latent_index ) * \
        LF_B->at(indices_B->at(index_B), latent_index );
  }

  return sum;
}
// Compute for update B
inline float ThreeMatrixFactorTrainer::ComputeProductPsiPsiAC(int thread_id, int frame_id, int index_A, int latent_index, MatrixBlob *LF_A, MatrixBlob *LF_C, std::vector<int> *indices_C ) {

  float sum = 0.;
  Settings *Settings_ = this->Settings_;

  // Note that we skip the sentinel value at the end!
  for (int index_C = 0; index_C < indices_C->size() - 1; ++index_C)
  {
    SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S3
    if (indices_C->at(index_C) < 0) continue;

    sum +=  LF_A->at(index_A, latent_index ) * \
        LF_C->at(indices_C->at(index_C), latent_index );
  }

  return sum;
}
// Compute for update A
inline float ThreeMatrixFactorTrainer::ComputeProductPsiPsiBC(int thread_id, int frame_id, int latent_index, MatrixBlob *LF_B, MatrixBlob *LF_C, std::vector<int> *indices_B, std::vector<int> *indices_C ) {

  float sum = 0.;
  Settings *Settings_ = this->Settings_;

  // Note that we skip the sentinel value at the end!
  for (int index_B = 0; index_B < indices_B->size() - 1; ++index_B)
  {
    SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S3
    if (indices_B->at(index_B) < 0) continue;

    // Note that we skip the sentinel value at the end!
    for (int index_C = 0; index_C < indices_C->size() - 1; ++index_C)
    {
      SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S3
      if (indices_C->at(index_C) < 0) continue;

      sum +=  LF_B->at(indices_B->at(index_B), latent_index ) * \
          LF_C->at(indices_C->at(index_C), latent_index );
    }
  }

  return sum;
}
inline float ThreeMatrixFactorTrainer::ComputeProductSC(int thread_id,
  int frame_id,
  int index_A,
  int index_B,
  int latent_index,
  Tensor3Blob *Sparse_S,
  MatrixBlob *LF_A,
  MatrixBlob *LF_B,
  MatrixBlob *LF_C
  ) {

  float sum = 0.;
  Settings *Settings_ = this->Settings_;

  for (int index_C = 0; index_C < Settings_->StageThree_Dimension_C; ++index_C)
  {
    sum += Sparse_S->at(index_A, index_B, index_C) * LF_C->at(index_C, latent_index);
  }

  return sum;
}
inline float ThreeMatrixFactorTrainer::ComputeProductSB(int thread_id,
  int frame_id,
  int index_A,
  int index_C,
  int latent_index,
  Tensor3Blob *Sparse_S,
  MatrixBlob *LF_A,
  MatrixBlob *LF_B,
  MatrixBlob *LF_C
  ) {

  float sum = 0.;
  Settings *Settings_ = this->Settings_;

  for (int index_B = 0; index_B < Settings_->StageThree_Dimension_B; ++index_B)
  {
    sum += Sparse_S->at(index_A, index_B, index_C) * LF_B->at(index_B, latent_index);
  }

  return sum;
}
inline float ThreeMatrixFactorTrainer::ComputeProductSBC(
                      int thread_id,
                      int index_A,
                      int latent_index,
                      Tensor3Blob * S,
                      MatrixBlob * B,
                      MatrixBlob * C) {
  float sum = 0.;

  // Check domain of contracted indices is equally large
  assert (S->rows == B->rows);
  assert (S->columns == C->rows);

  for (int index_B = 0; index_B < B->columns; ++index_B)
  {
    for (int index_C = 0; index_C < C->columns; ++index_C)
    {
      sum += S->at(index_A, index_B, index_C) * B->at(index_B, latent_index) * C->at(index_C, latent_index);
    }
  }

  return sum;
}

// Compute for Score
inline float ThreeMatrixFactorTrainer::ComputeProductPsiPsiS(int thread_id, int frame_id, int index_A, Tensor3Blob *Sparse_S, std::vector<int> *indices_B, std::vector<int> *indices_C ) {

  float sum = 0.;

  Settings *Settings_ = this->Settings_;

  for (int index_B = 0; index_B < indices_B->size() - 1; ++index_B)
  {
    SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S3
    if (indices_B->at(index_B) < 0) continue;

    for (int index_C = 0; index_C < indices_C->size() - 1; ++index_C)
    {
      SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S3
      if (indices_C->at(index_C) < 0) continue;

      sum += Sparse_S->at(index_A, indices_B->at(index_B), indices_C->at(index_C));
    }
  }

  return sum;
}
// Compute for Score
inline float ThreeMatrixFactorTrainer::ComputeProductPsiPsiABC(int thread_id, int frame_id, int index_A, MatrixBlob *LF_A, MatrixBlob *LF_B, MatrixBlob *SLF_B, MatrixBlob *LF_C, std::vector<int> *indices_B, std::vector<int> *indices_C) {


  Settings *Settings_ = this->Settings_;

  assert(LF_A->columns == LF_B->columns);
  if (Settings_->StageThree_UseSparseLatent_B_Term == 1) assert(LF_A->columns == SLF_B->columns);
  assert(LF_A->columns == LF_C->columns);

  // for (int index_B = 0; index_B < Settings_->StageThree_NumberOfNonZeroEntries_B; ++index_B)
  // {
  //   cout << "Blob_B_index: " << indices_B->at(index_B) << endl;
  // }

  // for (int index_C = 0; index_C < Settings_->DummyMultiplier * Settings_->StageThree_NumberOfNonZeroEntries_C; ++index_C)
  // {
  //   cout << "Blob_C_index:  " << indices_C->at(index_C) << endl;
  // }

  float sum = 0.;
  float Bterms = 0.;

  for (int index_B = 0; index_B < indices_B->size() - 1; ++index_B)
  {
    SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S3
    if (indices_B->at(index_B) < 0) continue;

    for (int index_C = 0; index_C < indices_C->size() - 1; ++index_C)
    {
      SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S3
      if (indices_C->at(index_C) < 0) continue;

      // if (frame_id % 1000000 == 0) {
      //   cout << "index_A: " << index_A << endl;
      //   cout << "ind_B  : " << indices_B->at(index_B) << endl;
      //   cout << "ind_C  : " << indices_C->at(index_C) << endl;
      // }

      for (int latent_index = 0; latent_index < LF_A->columns; ++latent_index)
      {
        Bterms = 0.;
        Bterms += LF_B->at(indices_B->at(index_B), latent_index );

        // Add the SPARSE latent factor B
        if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
          Bterms += SLF_B->at(indices_B->at(index_B), latent_index );
        }

        sum +=  LF_A->at(index_A, latent_index ) * \
            Bterms * \
            LF_C->at(indices_C->at(index_C), latent_index );

        // if (frame_id % 10000 == 0 and index_B == 0 and index_C == 0 and latent_index % 2 == 0){
        //   // cout << "thread_: " << thread_id << endl;
        //   cout << "latent_: " << latent_index << endl;
        //   cout << "LF_A   : " << LF_A->at(index_A, latent_index ) << endl;
        //   cout << "LF_B   : " << LF_B->at(indices_B->at(index_B), latent_index ) << endl;
        //   cout << "Bterms : " << Bterms << endl;
        //   // cout << "SLF_B  : " << SLF_B->at(indices_B->at(index_B), latent_index ) << endl;
        //   cout << "LF_C   : " << LF_C->at(indices_C->at(index_C), latent_index ) << endl;
        //   cout << "ABC  : " << sum << endl;
        //   cout << endl;
        // }
      }

    }
  }

  // if (frame_id % 10000 == 0){
  //   for (int index_B = 0; index_B < Settings_->StageThree_NumberOfNonZeroEntries_B; ++index_B){
  //     for (int index_C = 0; index_C < Settings_->DummyMultiplier * Settings_->StageThree_NumberOfNonZeroEntries_C; ++index_C){
  //       for (int latent_index = 0; latent_index < LF_A->columns; ++latent_index)
  //       {
  //         cout << "F " << frame_id << " LF_A LF_B LF_C " << " L: " << latent_index << " " << LF_A->at(index_A, latent_index ) << " " << LF_B->at(index_B, latent_index ) << " " << LF_C->at(index_C, latent_index ) << endl;
  //         // cout << "SLF_B  : " << SLF_B->at(indices_B->at(index_B), latent_index ) << endl;
  //       }
  //     }
  //   }
  //   cout << "ABC  : " << sum << endl;
  // }

  return sum;
}

void displayVector(std::vector<int> *v) {
  cout << "Contents of v: ";
  for (int i = 0; i < v->size(); ++i)
  {
    cout << v->at(i) << " ";
  }
  cout << endl;
}

inline float ThreeMatrixFactorTrainer::ComputeScore(int thread_id, int index_A, int frame_id) {
  Settings  *Settings_ = this->Settings_;
  MatrixBlob  *LF_A    = &this->LF_A;
  MatrixBlob  *LF_B    = &this->LF_B;
  MatrixBlob  *SLF_B   = &this->SLF_B;
  MatrixBlob  *LF_C    = &this->LF_C;
  Tensor3Blob *Sparse_S  = &this->Sparse_S;
  VectorBlob  *Bias_A  = &this->Bias_A;

  float temp, temp2, temp3, temp4;
  float score = 0.;

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);

  score += this->ComputeProductPsiPsiABC(thread_id, frame_id, index_A, LF_A, LF_B, SLF_B, LF_C, &indices_B, &indices_C );
  temp = score;

  if (Settings_->StageThree_UseSparseMatS == 1) score += this->ComputeProductPsiPsiS(thread_id, frame_id, index_A, Sparse_S, &indices_B, &indices_C );
  temp2 = score;

  if (thread_id == 0 and Settings_->EnableDebugPrinter_Level3 == 1 and frame_id % 1000000 == 0) displayVector(&indices_B);

  if (Settings_->StageThree_UseSpatialRegularization == 1){
    std::vector<int> indices_BB = this->getIndices_B(frame_id, 31);
    std::vector<int> indices_CC = this->getIndices_C(frame_id, 31);

    if (thread_id == 0 and Settings_->EnableDebugPrinter_Level3 == 1 and frame_id % 10000 == 0) displayVector(&indices_BB);

    score += this->ComputeProductPsiPsiABC(thread_id, frame_id, index_A, LF_A, LF_B, SLF_B, LF_C, &indices_BB, &indices_CC );
    temp3 = score;
    if (thread_id == 0 and Settings_->StageThree_UseSparseMatS == 1) score += this->ComputeProductPsiPsiS(thread_id, frame_id, index_A, Sparse_S, &indices_BB, &indices_CC );
    temp4 = score;
  }

  // Add the bias
  score += Bias_A->data[index_A];

  if (thread_id == 0 and Settings_->EnableDebugPrinter_Level2 == 1 and frame_id % 10000 == 0) {
    if (temp2 - temp >= 0.0) {

      cout << endl;
      PrintDelimiter(0, 0, 80, '=');
      cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "F " << std::setw(9) << frame_id << " : ABC   " << std::setw(PRINT_FLOAT_PRECISION_SCORE+1) << temp << endl;
      cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "F " << std::setw(9) << frame_id << " : ABC+S   " << std::setw(PRINT_FLOAT_PRECISION_SCORE+1) << temp2 << " delta: " << temp2 - temp << endl;
      if (Settings_->StageThree_UseSpatialRegularization == 1) cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "F " << std::setw(9) << frame_id << " : ABC+S 2 " << std::setw(PRINT_FLOAT_PRECISION_SCORE+1) << temp3 << " delta: " << temp3 - temp2 << endl;
      if (Settings_->StageThree_UseSpatialRegularization == 1) cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "F " << std::setw(9) << frame_id << " : ABC+S 2 " << std::setw(PRINT_FLOAT_PRECISION_SCORE+1) << temp4 << " delta: " << temp4 - temp3 << endl;
      cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "F " << std::setw(9) << frame_id << " : ABC+S+B " << std::setw(PRINT_FLOAT_PRECISION_SCORE+1) << score << " delta: " << score - temp2 << endl;
      PrintDelimiter(0, 0, 80, '=');
      cout << endl;
    }
  }


  return score;
}

inline float Trainer::cap(int thread_id, int index_A, int frame_id, float score, float cap_hi, float cap_lo) {
  Settings  *Settings_    = this->Settings_;
  if (score > cap_hi) {
    if (thread_id == 0 and frame_id % 10000 == 0) {
      PrintFancy(Settings_->session_start_time, "Blowup! Task: " + to_string(index_A) + " Frame: " + to_string(frame_id) + " Score: " + to_string(score) + " -- capping to: " + to_string(SCORE_BLOWUP_THRESHOLD_ABOVE));
    }
    return cap_hi;
  } else if (score < cap_lo) {
    if (thread_id == 0 and frame_id % 10000 == 0) {
      PrintFancy(Settings_->session_start_time, "Blowup! Task: " + to_string(index_A) + " Frame: " + to_string(frame_id) + " Score: " + to_string(score) + " -- capping to: " + to_string(SCORE_BLOWUP_THRESHOLD_BELOW));
    }
    return cap_lo;
  } else {
    return score;
  }
}

inline int ThreeMatrixFactorTrainer::ComputeConditionalScore(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeConditionalScore");

  Settings  *Settings_    = this->Settings_;
  MatrixBlob  *ConditionalScore = &(this->ConditionalScore);

  float score = this->ComputeScore(thread_id, index_A, frame_id);

  if (Settings_->EnableDebugPrinter_Level3 == 1 and thread_id == 0 and frame_id % 10000 == 0) {
    cout << "[Debug] ";
    if (index_A == 8 or index_A == 9) cout << "COPULATION! ";
    cout << setw(10)  << "Task: " << index_A;
    cout << setw(10) << " F: " << frame_id;
    cout << setw(4)  << " G: " << ground_truth_label;
    cout << setw(10) << setprecision(6) << " S: " << score;
    cout << endl;
  }

  if (Settings_->CapScores == 1) {
    score = cap(thread_id, index_A, frame_id, score, SCORE_BLOWUP_THRESHOLD_ABOVE, SCORE_BLOWUP_THRESHOLD_BELOW);
  } else {
    if (this->CheckIfFloatIsNan(score, "") || score > SCORE_BLOWUP_THRESHOLD_ABOVE || score < SCORE_BLOWUP_THRESHOLD_BELOW) {
      PrintFancy(Settings_->session_start_time, "Blowup! Task: " + to_string(index_A) + " Frame: " + to_string(frame_id) + " Score: " + to_string(score));
      global_reset_training = 1;
      return -1;
    }
  }

  *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)) = score;

  return 0;
}

// Compute for A
inline void ThreeMatrixFactorTrainer::ComputeLF_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeLF_A_Update");

  Settings  *Settings_     = this->Settings_;
  MatrixBlob  *ConditionalScore  = &(this->ConditionalScore);

  Tensor3Blob *Sparse_S      = &(this->Sparse_S);

  MatrixBlob  *LF_A        = &(this->LF_A);
  MatrixBlob  *LF_A_diff     = &(this->LF_A_Diff);
  MatrixBlob  *LF_A_DiffSq_Cache = &(this->LF_A_DiffSq_Cache);

  MatrixBlob *LF_B         = &this->LF_B;
  MatrixBlob *LF_C         = &this->LF_C;

  float ExponentialScore     = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float learning_rate      = Settings_->StageThree_CurrentLearningRate;
  float reg_LF_A         = Settings_->StageThree_Regularization_LF_A;
  int   n_MiniBatchSize      = Settings_->StageThree_MiniBatchSize;

  float LF_A_update        = 0.;
  float LF_A_old         = 0.;
  float prodPsiPsiBC       = 0.;
  float strongweakLF_A       = 0.;
  float factor           = 0.;

  float reg_S_BC         = Settings_->StageThree_Regularization_RepulsionTerm;
  float product_SBC        = 0.;

  float gradient_reg       = 0.;
  float grad_at_zero       = 0.;

  if (ground_truth_label == 0) {
    strongweakLF_A = Settings_->LossWeightWeak;
    factor     = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakLF_A = Settings_->LossWeightStrong;
    factor     = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);


  assert (LF_A->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  for (int latent_index = 0; latent_index < LF_A->columns; ++latent_index) {

    LF_A_update = 0.0;
    prodPsiPsiBC = 0.0;

    LF_A_old = LF_A->at(index_A, latent_index);

    // Indicator feature --> we only select one column of entries of W!
    prodPsiPsiBC = ComputeProductPsiPsiBC(thread_id,
                      frame_id,
                      latent_index,
                      LF_B,
                      LF_C,
                      &indices_B,
                      &indices_C
                      );

    LF_A_update += strongweakLF_A * factor * prodPsiPsiBC / n_MiniBatchSize;

    // Regularization
    if (latent_index < Settings_->StageThree_SubDimension_1) {

      if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
      else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

      // L1 regularization
      if (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3) {

        LF_A_update += Settings_->StageThree_A_RegularizationStrength_1_Sparse * this->GetLassoGradient(thread_id, LF_A_old) / n_MiniBatchSize;

      }

      // L2 regularization
      if (Settings_->StageThree_RegularizationType_1 == 2 or Settings_->StageThree_RegularizationType_1 == 3) {

        LF_A_update += Settings_->StageThree_A_RegularizationStrength_1_Dense * LF_A_old / n_MiniBatchSize;

      }
    } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

      // cout << latent_index << " computing update LF_A -- subdim2" << endl;


      if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
      else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

      // L1 regularization
      if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {

        LF_A_update += Settings_->StageThree_A_RegularizationStrength_2_Sparse * this->GetLassoGradient(thread_id, LF_A_old) / n_MiniBatchSize;

      }

      // L2 regularization
      if (Settings_->StageThree_RegularizationType_2 == 2 or Settings_->StageThree_RegularizationType_2 == 3) {

        LF_A_update += Settings_->StageThree_A_RegularizationStrength_2_Dense * LF_A_old / n_MiniBatchSize;

      }

      // if (thread_id == 0 and index_A == 0 and latent_index % 10 == 0) cout << "[Debug] Latent index: " << latent_index << " -- computed LF_A_update: " << LF_A_update << endl;

    };

    // S, ABC repulsion term gradient
    if (Settings_->StageThree_UseSparseMatS == 1 and Settings_->StageThree_UseRepulsion_S_ABC == 1) {
      assert (Settings_->StageThree_UseRepulsion_S_BC == 0);
      product_SBC = ComputeProductSBC(thread_id,
                      index_A,
                      latent_index,
                      Sparse_S,
                      LF_B,
                      LF_C);

      LF_A_update += reg_S_BC * product_SBC / n_MiniBatchSize;
    }

    // Apply update
    *(LF_A_diff->att(index_A, latent_index)) += LF_A_update;
  }
}
inline void ThreeMatrixFactorTrainer::ProcessLF_A_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessLF_A_Updates");
  Settings      *Settings_     = this->Settings_;
  MatrixBlob      *LF_A        = &(this->LF_A);
  MatrixBlob      *LF_A_diff     = &(this->LF_A_Diff);
  MatrixBlob      *LF_A_DiffSq_Cache = &(this->LF_A_DiffSq_Cache);
  int         n_MiniBatchSize  = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der    = 0.;
  float LF_A_update      = 0.;
  float adaptive_learning_rate = 0.;

  assert(LF_A->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  int sign = 0;

  for (int latent_index = 0; latent_index < LF_A->columns; ++latent_index) {

    // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
    // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
    adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate;
    //ADAPTIVE_LEARNING_RATE
    if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
      cum_sum_first_der     = LF_A_DiffSq_Cache->at(index_A, latent_index);
      adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);

      if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
        // PrintFancy(Settings_->session_start_time, "Learning rate update clamped! A: " + to_string(index_A));
        adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
      }
    }

    LF_A_update = adaptive_learning_rate * LF_A_diff->at(index_A, latent_index);

    if (latent_index < Settings_->StageThree_SubDimension_1) {

      if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
      else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

      if (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3) {
        *(LF_A->att(index_A, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_A->at(index_A, latent_index), LF_A_update);
      }
      else if (Settings_->StageThree_RegularizationType_1 == 2) {
        *(LF_A->att(index_A, latent_index)) -= LF_A_update;
      }
    } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

      if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
      else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

      if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {
        *(LF_A->att(index_A, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_A->at(index_A, latent_index), LF_A_update);
      }
      else if (Settings_->StageThree_RegularizationType_2 == 2) {
        *(LF_A->att(index_A, latent_index)) -= LF_A_update;
      }
    };

    // *(LF_A->att(index_A, latent_index)) = SoftThresholdUpdate(LF_A->at(index_A, latent_index), LF_A_update);

  }
}

// Compute for B
inline void ThreeMatrixFactorTrainer::ComputeLF_B_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  Settings  *Settings_     = this->Settings_;

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);
  this->ComputeLF_B_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_B, &indices_C, 1);

  // Load spatial regularization / sliding window indices
  if (Settings_->StageThree_UseSpatialRegularization == 1){
    if (Settings_->EnableDebugPrinter_Level3 == 1 and thread_id == 0 and frame_id % 1000000 == 0) printf("task: %i f: %i g: %i starting spatial regularization -- 2x2 sliding window features \n", index_A, frame_id, ground_truth_label);
    std::vector<int> indices_BB = this->getIndices_B(frame_id, 31);
    std::vector<int> indices_CC = this->getIndices_C(frame_id, 31);
    this->ComputeLF_B_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_BB, &indices_CC, 2);
  }
}
inline void ThreeMatrixFactorTrainer::ComputeLF_B_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel) {

  if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 500000 == 0) {
    PrintFancy(Settings_->session_start_time, "F " + to_string(frame_id) + "ComputeLF_B -- SpatRegLevel " + to_string(SpatRegLevel));
  }

  Settings  *Settings_     = this->Settings_;
  MatrixBlob  *ConditionalScore  = &(this->ConditionalScore);

  MatrixBlob  *LF_A        = &(this->LF_A);
  MatrixBlob  *LF_B        = &(this->LF_B);
  MatrixBlob  *LF_B_Diff     = &(this->LF_B_Diff);
  MatrixBlob  *LF_B_DiffSq_Cache = &(this->LF_B_DiffSq_Cache);
  MatrixBlob  *SLF_B       = &(this->SLF_B);
  MatrixBlob  *LF_C        = &(this->LF_C);
  Tensor3Blob *Sparse_S      = &(this->Sparse_S);

  float ExponentialScore     = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float learning_rate      = Settings_->StageThree_CurrentLearningRate;
  float SpatRegMultiplier    = 0.;
  int   n_MiniBatchSize      = Settings_->StageThree_MiniBatchSize;

  float LF_B_update        = 0.;
  float LF_B_old         = 0.;
  float prodPsiPsiAC       = 0.;
  float strongweakLF_B       = 0.;
  float factor           = 0.;
  int   compute_product      = 0;
  float gradient_reg       = 0.;
  float grad_at_zero       = 0.;

  float reg_repulsion_BB     = Settings_->StageThree_Regularization_repulsion_BB;

  assert (SpatRegLevel == 1 or SpatRegLevel == 2);
  if (SpatRegLevel == 1){
    SpatRegMultiplier = Settings_->StageThree_B_SpatReg_Multiplier_Level1;
  } else if (SpatRegLevel == 2){
    SpatRegMultiplier = Settings_->StageThree_B_SpatReg_Multiplier_Level2;
  }

  assert (ground_truth_label == 0 or ground_truth_label == 1);
  if (ground_truth_label == 0) {
    strongweakLF_B = Settings_->LossWeightWeak;
    factor     = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakLF_B = Settings_->LossWeightStrong;
    factor     = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  float index_B_val = 0.0;
  int index_B_ptr   = 0;
  float sum_B_bk  = 0.;

  float  temp     = 0.0;

  bool statement;

  assert( Settings_->StageThree_Dimension_B == LF_B->rows);
  assert( LF_B->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
  assert( Settings_->StageThree_TrainSubDimension_1 == 0 or Settings_->StageThree_TrainSubDimension_1 == 1);
  assert( Settings_->StageThree_TrainSubDimension_2 == 0 or Settings_->StageThree_TrainSubDimension_2 == 1);

  for (int index_B = 0; index_B < LF_B->rows; ++index_B) {

    // if (indices_B[index_B_ptr] < 0) break;

    if (index_B == indices_B->at(index_B_ptr)) {
      index_B_val = 1.0;
      compute_product = 1;
      index_B_ptr += 1;
    }
    else {
      index_B_val = 0.0;
      compute_product = 0;
    }

    for (int latent_index = 0; latent_index < LF_B->columns; ++latent_index) {

      LF_B_update  = 0.0;
      prodPsiPsiAC = 0.0;
      LF_B_old   = LF_B->at(index_B, latent_index);

      if (Settings_->StageThree_TrainSubDimension_1 == 1 or Settings_->StageThree_TrainSubDimension_2 == 1) {
        if (compute_product == 1){

          prodPsiPsiAC = ComputeProductPsiPsiAC(thread_id, frame_id, index_A, latent_index, LF_A, LF_C, indices_C);

          LF_B_update += strongweakLF_B * factor * index_B_val * prodPsiPsiAC / n_MiniBatchSize;

          temp = LF_B_update;
          if (statement) printf("DEBUG - Gr -- f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i LF_Bu %8.5e \n", frame_id, index_A, index_B, latent_index, indices_B->at(index_B_ptr), LF_B_update);

        }
      }

      if (latent_index < Settings_->StageThree_SubDimension_1) {

        if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

        statement = Settings_->EnableDebugPrinter_Level3 == 1 and (thread_id == 0 and (frame_id % 1000) == 0 and ((latent_index == 0) or (latent_index == Settings_->StageThree_SubDimension_1)) and (index_B == indices_B->at(0)) and (index_B_ptr > 0));

        // L1 regularization
        if (Settings_->StageThree_RegularizationType_1 == 1 || Settings_->StageThree_RegularizationType_1 == 3) {

          temp = LF_B_update;
          LF_B_update += Settings_->StageThree_B_RegularizationStrength_1_Sparse * this->GetLassoGradient(thread_id, LF_B_old) / n_MiniBatchSize;
          if (statement) printf("DEBUG - L1 -- f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i LF_B_update %8.5e delta %8.5e \n", frame_id, index_A, index_B, latent_index, indices_B->at(index_B_ptr), LF_B_update, LF_B_update - temp);
        }

        // L2 regularization
        if (Settings_->StageThree_RegularizationType_1 == 2 || Settings_->StageThree_RegularizationType_1 == 3) {

          temp = LF_B_update;
          LF_B_update += Settings_->StageThree_B_RegularizationStrength_1_Dense * LF_B_old / n_MiniBatchSize;
          if (statement) printf("DEBUG - L2 -- f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i LF_B_update %8.5e delta %8.5e \n", frame_id, index_A, index_B, latent_index, indices_B->at(index_B_ptr), LF_B_update, LF_B_update - temp);
        }

      } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

        if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

        // Add repulsive term -- loss += lambda sum_kk' sum_b B_bk Btilde_bk'
        // if (Settings_->StageThree_UseRepulsion_S_ABC == 1) {
        //   // if (thread_id == 0 and index_A == 0 and index_B == 0 and latent_index % 10 == 0) cout << std::setprecision(20) << "[Debug] Using repulsive term for Btilde_bk' -- " << frame_id << " " << latent_index << " LF_B_update so far: " << LF_B_update << endl;

        //   for (int l_index = 0; l_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2; ++l_index)
        //   {
        //     if (l_index == latent_index) continue;
        //     LF_B_update += Settings_->StageThree_Regularization_S_BC * LF_B->at(index_B, l_index) / n_MiniBatchSize;
        //   }
        // }

        statement = Settings_->EnableDebugPrinter_Level3 == 1 and (thread_id == 0 and (frame_id % 1000) == 0 and ((latent_index == 0) or (latent_index == Settings_->StageThree_SubDimension_1)) and (index_B == indices_B->at(0)) and (index_B_ptr > 0));

        if (statement and SpatRegLevel == 2) printf("DEBUG - f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i Settings: subdim2 reg-type: %i trainsubdim2: %i SpatRegLevel: %i\n", frame_id, index_A, index_B, latent_index, indices_B->at(0), Settings_->StageThree_RegularizationType_2, Settings_->StageThree_TrainSubDimension_2, SpatRegLevel);

        // L1 regularization
        if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {
          temp = LF_B_update;
          LF_B_update += Settings_->StageThree_B_RegularizationStrength_2_Sparse * this->GetLassoGradient(thread_id, LF_B_old) / n_MiniBatchSize;
          if (statement) printf("DEBUG - L1 -- f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i LF_B_update %8.5e delta %8.5e \n", frame_id, index_A, index_B, latent_index, indices_B->at(index_B_ptr), LF_B_update, LF_B_update - temp);
        }

        // L2 regularization
        if (Settings_->StageThree_RegularizationType_2 == 2 or Settings_->StageThree_RegularizationType_2 == 3) {
          temp = LF_B_update;
          LF_B_update += Settings_->StageThree_B_RegularizationStrength_2_Dense * LF_B_old / n_MiniBatchSize;
          if (statement) printf("DEBUG - L2 -- f: %10.8i task: %4.2i b: %4.2i l: %4.2i %4.2i LF_B_update %8.5e delta %8.5e \n", frame_id, index_A, index_B, latent_index, indices_B->at(index_B_ptr), LF_B_update, LF_B_update - temp);
        }
      }

      if (statement) cout << endl;

      // S, ABC repulsion term gradient
      if (Settings_->StageThree_UseSparseMatS == 1 and Settings_->StageThree_UseRepulsion_S_ABC == 1) {
        assert (Settings_->StageThree_UseRepulsion_S_BC == 0);
        LF_B_update += Settings_->StageThree_Regularization_S_BC * ComputeProductSBC(thread_id, index_B, latent_index, Sparse_S, LF_B, LF_C) / n_MiniBatchSize;
      }

      // Term from the LF_B, SLF_B repulsion term in our loss
      if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
        LF_B_update += reg_repulsion_BB * SLF_B->at(index_B, latent_index) / n_MiniBatchSize;
      }

      if (thread_id == 0 and index_B == indices_B->at(index_B_ptr)) cout << setw(10) << "DEBUG - LF_B update - index_B: " << indices_B->at(index_B_ptr) << " LF_B_update " << LF_B_update << endl;

      *(LF_B_Diff->att(index_B, latent_index)) += SpatRegMultiplier * LF_B_update;



      // DEBUG!!
      if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 10000 == 0){
        if (thread_id == 0 and index_B % 100 == 0){
          cout << "DEBUG -- F " << frame_id << " B "  << index_B << " L " << latent_index << " prodPsiPsiAC: " << prodPsiPsiAC << " LF_B_update: " << LF_B_update << " LF_B_Diff->at(index_B, latent_index) " << *(LF_B_Diff->att(index_B, latent_index)) << endl;
        }
      }

      // DEBUG!!
      if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 10000 == 0 and (abs(LF_B_update) > 10.0 or abs(*(LF_B_Diff->att(index_B, latent_index))) > 100.0 or abs(*(LF_B->att(index_B, latent_index))) > 100.0)){
        if (thread_id == 0 and index_B % 100 == 0){
          cout << "DEBUG -- Blow-up -- F " << frame_id << " B "  << index_B << " L " << latent_index << " prodPsiPsiAC: " << prodPsiPsiAC << " LF_B_update: " << LF_B_update << " LF_B_Diff->at(index_B, latent_index) " << *(LF_B_Diff->att(index_B, latent_index)) << endl;
        }
      }

    }
  }
}
inline void ThreeMatrixFactorTrainer::ProcessLF_B_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessLF_A_Updates");
  Settings   *Settings_     = this->Settings_;
  MatrixBlob *LF_B        = &(this->LF_B);
  MatrixBlob *LF_B_Diff     = &(this->LF_B_Diff);
  MatrixBlob *LF_B_DiffSq_Cache = &(this->LF_B_DiffSq_Cache);

  int n_MiniBatchSize      = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der    = 0.;
  float LF_B_update      = 0.;
  float adaptive_learning_rate = 0.;

  assert(LF_B->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  int sign = 0;

  for (int index_B = 0; index_B < LF_B->rows; ++index_B) {
    for (int latent_index = 0; latent_index < LF_B->columns; ++latent_index) {

      // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
      // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
      adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate;

      //ADAPTIVE_LEARNING_RATE
      if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
        cum_sum_first_der     = LF_B_DiffSq_Cache->at(index_B, latent_index);
        adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);

        if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
          adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
        }
      }

      LF_B_update = adaptive_learning_rate * LF_B_Diff->at(index_B, latent_index);

      if (latent_index < Settings_->StageThree_SubDimension_1) {

        if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

        if (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3) {
          *(LF_B->att(index_B, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_B->at(index_B, latent_index), LF_B_update);
        }
        else if (Settings_->StageThree_RegularizationType_1 == 2) {
          *(LF_B->att(index_B, latent_index)) -= LF_B_update;
        }

      } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

        if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

        if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {
          *(LF_B->att(index_B, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_B->at(index_B, latent_index), LF_B_update);
        }
        else if (Settings_->StageThree_RegularizationType_2 == 2) {
          *(LF_B->att(index_B, latent_index)) -= LF_B_update;
        }
      }

    }
  }
}

// Compute for B sparse
inline void ThreeMatrixFactorTrainer::ComputeSLF_B_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeLF_A_Update");

  Settings  *Settings_      = this->Settings_;
  MatrixBlob  *ConditionalScore   = &(this->ConditionalScore);

  MatrixBlob  *LF_A         = &(this->LF_A);
  MatrixBlob  *LF_B         = &(this->LF_B);
  MatrixBlob  *SLF_B        = &(this->SLF_B);
  MatrixBlob  *SLF_B_Diff     = &(this->SLF_B_Diff);
  MatrixBlob  *SLF_B_DiffSq_Cache = &(this->SLF_B_DiffSq_Cache);
  MatrixBlob  *LF_C         = &(this->LF_C);

  float ExponentialScore      = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float learning_rate       = Settings_->StageThree_CurrentLearningRate;
  float reg_SLF_B         = Settings_->StageThree_Regularization_SLF_B_Level1;
  float reg_repulsion_BB      = Settings_->StageThree_Regularization_repulsion_BB;
  int   n_MiniBatchSize       = Settings_->StageThree_MiniBatchSize;

  float SLF_B_update        = 0.;
  float old_SLF_B         = 0.;
  float prodPsiPsiAC        = 0.;
  float strongweakSLF_B       = 0.;
  float factor          = 0.;
  float gradient_reg        = 0.;
  float grad_at_zero        = 0.;

  int level2start         = Settings_->StageThree_Dimension_B;

  if (ground_truth_label == 0) {
    strongweakSLF_B = Settings_->LossWeightWeak;
    factor      = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakSLF_B = Settings_->LossWeightStrong;
    factor      = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);

  float index_B_val = 0.0;
  int index_B_ptr   = 0;

  for (int index_B = 0; index_B < SLF_B->rows; ++index_B) {

    if (indices_B[index_B_ptr] < 0) break;

    if (index_B == indices_B[index_B_ptr]) {
      index_B_val = 1.0;
      index_B_ptr += 1;
    }
    else {
      index_B_val = 0.0;
    }

    if (index_B == 0)      reg_SLF_B = Settings_->StageThree_Regularization_SLF_B_Level1;
    if (index_B == level2start)  reg_SLF_B = Settings_->StageThree_Regularization_SLF_B_Level2;

    for (int latent_index = 0; latent_index < SLF_B->columns; ++latent_index) {

      SLF_B_update = 0.0;

      prodPsiPsiAC = ComputeProductPsiPsiAC( thread_id,
                        frame_id,
                        index_A,
                        latent_index,
                        LF_A,
                        LF_C,
                        &indices_C
                        );

      // Gradient from the loss function
      SLF_B_update += strongweakSLF_B * factor * index_B_val * prodPsiPsiAC / n_MiniBatchSize;

      // Choose 3 for L1 + L2
      // Add the L1 regularization
      if (Settings_->StageThree_SLF_B_RegularizationType == 1 or Settings_->StageThree_SLF_B_RegularizationType == 3) {

        if (Settings_->GradientRandomizeAtZero == 1)  grad_at_zero = 0.0; // giveRandom(0.0, 1.0);

        SLF_B_update += reg_SLF_B * this->GetLassoGradient(thread_id, old_SLF_B) / n_MiniBatchSize;
      }

      // Add the L2 regularization
      if (Settings_->StageThree_SLF_B_RegularizationType == 2 or Settings_->StageThree_SLF_B_RegularizationType == 3) {
        SLF_B_update += reg_SLF_B * SLF_B->at(index_B, latent_index) / n_MiniBatchSize;
      }

      // Term from the LF_B, SLF_B repulsion term in our loss
      if (Settings_->StageThree_UseSparseLatent_B_Term == 1) {
        SLF_B_update += reg_repulsion_BB * LF_B->at(index_B, latent_index) / n_MiniBatchSize;
      }

      *(SLF_B_Diff->att(index_B, latent_index)) += SLF_B_update;
    }
  }
}
inline void ThreeMatrixFactorTrainer::ProcessSLF_B_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessLF_A_Updates");
  Settings  *Settings_      = this->Settings_;
  MatrixBlob  *SLF_B        = &(this->SLF_B);
  MatrixBlob  *SLF_B_Diff     = &(this->SLF_B_Diff);
  MatrixBlob  *SLF_B_DiffSq_Cache = &(this->SLF_B_DiffSq_Cache);

  int n_MiniBatchSize       = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der     = 0.;
  float SLF_B_update        = 0.;
  float adaptive_learning_rate  = 0.;

  for (int index_B = 0; index_B < SLF_B->rows; ++index_B) {
    for (int latent_index = 0; latent_index < SLF_B->columns; ++latent_index) {

      // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
      // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
      adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate;

      //ADAPTIVE_LEARNING_RATE
      if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
        cum_sum_first_der     = SLF_B_DiffSq_Cache->at(index_B, latent_index);
        adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);

        if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
          adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
        }
      }

      SLF_B_update             = adaptive_learning_rate * SLF_B_Diff->at(index_B, latent_index);
      *(SLF_B->att(index_B, latent_index)) -= SLF_B_update;

      if (SLF_B->at(index_B, latent_index) < 0.0) {
        *(SLF_B->att(index_B, latent_index)) = 0.0;
      }
    }
  }
}

// Compute for C
inline void ThreeMatrixFactorTrainer::ComputeLF_C_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {

  Settings  *Settings_   = this->Settings_;

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);
  this->ComputeLF_C_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_B, &indices_C, 1);

  // Load spatial regularization / sliding window indices
  if (Settings_->StageThree_UseSpatialRegularization == 1){
    std::vector<int> indices_BB = this->getIndices_B(frame_id, 31);
    std::vector<int> indices_CC = this->getIndices_C(frame_id, 31);

    this->ComputeLF_C_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_BB, &indices_CC, 2);
  }
}
inline int getIndexFirstNonNegative(std::vector<int> *indices_C){
  // We sorted the defender occupancy feature already. Now look for first defender location that is not -1

  int index_C_ptr = 0;

  while (1) {
    if (index_C_ptr >= indices_C->size()){
      break;
    };
    if (indices_C->at(index_C_ptr) < 0) {
      index_C_ptr += 1;
    } else {
      return index_C_ptr;
    }
  }
  return index_C_ptr - 1;
}

inline void ThreeMatrixFactorTrainer::ComputeLF_C_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel) {

  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeLF_C_Update");
  if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 500000 == 0) {
    PrintFancy(Settings_->session_start_time, "F " + to_string(frame_id) + "ComputeLF_C -- SpatRegLevel " + to_string(SpatRegLevel));
  }

  Settings  *Settings_     = this->Settings_;
  MatrixBlob  *ConditionalScore  = &(this->ConditionalScore);
  Tensor3Blob *Sparse_S      = &(this->Sparse_S);
  MatrixBlob  *LF_A        = &(this->LF_A);
  MatrixBlob  *LF_B        = &(this->LF_B);
  MatrixBlob  *LF_C        = &(this->LF_C);
  MatrixBlob  *LF_C_Diff     = &(this->LF_C_Diff);
  MatrixBlob  *LF_C_DiffSq_Cache = &(this->LF_C_DiffSq_Cache);

  float ExponentialScore     = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float learning_rate      = Settings_->StageThree_CurrentLearningRate;
  int   n_MiniBatchSize      = Settings_->StageThree_MiniBatchSize;

  float SpatRegMultiplier    = 0.;
  float LF_C_update        = 0.;
  float LF_C_old         = 0.;
  float prodPsiPsiAB       = 0.;
  float strongweakLF_C       = 0.;
  float factor           = 0.;

  int   compute_product      = 0;

  float gradient_reg       = 0.;
  float grad_at_zero       = 0.;

  assert (SpatRegLevel == 1 or SpatRegLevel == 2);
  if (SpatRegLevel == 1){
    SpatRegMultiplier = Settings_->StageThree_C_SpatReg_Multiplier_Level1;
  } else if (SpatRegLevel == 2){
    SpatRegMultiplier = Settings_->StageThree_C_SpatReg_Multiplier_Level2;
  }

  if (ground_truth_label == 0) {
    strongweakLF_C   = Settings_->LossWeightWeak;
    factor      = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakLF_C   = Settings_->LossWeightStrong;
    factor      = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  float index_C_val = 0.0;

  // We sorted the defender occupancy feature already. Now look for first defender location that is not -1
  int index_C_ptr   = getIndexFirstNonNegative(indices_C);

  // if (Settings_->EnableDebugPrinter_Level3 == 1 and thread_id == 0 and frame_id % 500000 == 0) {
  //   PrintFancy(Settings_->session_start_time, "F " + to_string(frame_id) + " -- Defender OutOfBounds detected. Old index_C: " + to_string(index_C + 1) + " --> " + to_string(index_C));
  // }

  // Loop over every defender grid cell
  for (int index_C = 0; index_C < LF_C->rows; ++index_C) {

    if (index_C == indices_C->at(index_C_ptr)) {
      index_C_val   = 1.0;
      compute_product = 1;
    } else {
      index_C_val = 0.0;
      compute_product = 0;
    }

    SKIP_OUT_OF_BOUNDS_INDICES_DEF_S3

    assert (LF_C->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

    for (int latent_index = 0; latent_index < LF_C->columns; ++latent_index) {

      LF_C_update = 0.0;

      LF_C_old = LF_C->at(index_C, latent_index);

      // Indicator feature --> we only select one column of entries of W!
      if (compute_product == 1) {
        prodPsiPsiAB = ComputeProductPsiPsiAB( thread_id,
                          frame_id,
                          index_A,
                          latent_index,
                          LF_A,
                          LF_B,
                          indices_B
                          );

        LF_C_update += strongweakLF_C * factor * index_C_val * prodPsiPsiAB / n_MiniBatchSize;
      }

      // Regularization
      if (latent_index < Settings_->StageThree_SubDimension_1) {

        if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

        if (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3) {
          LF_C_update += Settings_->StageThree_C_RegularizationStrength_1_Sparse * this->GetLassoGradient(thread_id, LF_C_old) / n_MiniBatchSize;
        }

        if (Settings_->StageThree_RegularizationType_1 == 2 or Settings_->StageThree_RegularizationType_1 == 3) {
          LF_C_update += Settings_->StageThree_C_RegularizationStrength_1_Dense * LF_C_old / n_MiniBatchSize;
        }
      } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

        if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

        if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {
          LF_C_update += Settings_->StageThree_C_RegularizationStrength_2_Sparse * this->GetLassoGradient(thread_id, LF_C_old) / n_MiniBatchSize;
        }

        if (Settings_->StageThree_RegularizationType_2 == 2 or Settings_->StageThree_RegularizationType_2 == 3) {
          LF_C_update += Settings_->StageThree_C_RegularizationStrength_2_Dense * LF_C_old / n_MiniBatchSize;
        }
      }

      *(LF_C_Diff->att(index_C, latent_index)) += SpatRegMultiplier * LF_C_update;
    }

    // If we hit the current index, go to the next one
    if (index_C == indices_C->at(index_C_ptr)){
      index_C_ptr += 1;
    }
    // If we find duplicate index ahead, we keep index_C at its current position
    if (index_C == indices_C->at(index_C_ptr) and index_C_ptr < indices_C->size()-1){
      if (indices_C->at(index_C_ptr - 1) == indices_C->at(index_C_ptr)) {
        index_C -= 1;
      }
    }

  }
}
inline void ThreeMatrixFactorTrainer::ProcessLF_C_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessLF_C_Updates");
  Settings      *Settings_     = this->Settings_;
  MatrixBlob      *LF_C        = &(this->LF_C);
  MatrixBlob      *LF_C_Diff     = &(this->LF_C_Diff);
  MatrixBlob      *LF_C_DiffSq_Cache = &(this->LF_C_DiffSq_Cache);
  int         nOccupancyFeatures = Settings_->StageThree_Dimension_B;
  int         n_MiniBatchSize  = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der        = 0.;
  float LF_C_update            = 0.;
  float adaptive_learning_rate       = 0.;

  assert(LF_C->columns == Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);

  for (int index_C = 0; index_C < LF_C->rows; ++index_C) {
    for (int latent_index = 0; latent_index < LF_C->columns; ++latent_index) {

      if (latent_index == 12) continue;

      // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
      // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
      adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate;

      //ADAPTIVE_LEARNING_RATE
      if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
        cum_sum_first_der     = LF_C_DiffSq_Cache->at(index_C, latent_index);
        adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);

        if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
          // PrintFancy(Settings_->session_start_time, "Learning rate update clamped! A: " + to_string(index_A));
          adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
        }
      }

      LF_C_update = adaptive_learning_rate * LF_C_Diff->at(index_C, latent_index);

      if (latent_index < Settings_->StageThree_SubDimension_1) {

        if (Settings_->StageThree_TrainSubDimension_1 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_1 == 1);

        if (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3) {
          *(LF_C->att(index_C, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_C->at(index_C, latent_index), LF_C_update);
        }
        else if (Settings_->StageThree_RegularizationType_1 == 2) {
          *(LF_C->att(index_C, latent_index)) -= LF_C_update;
        }
      } else if (latent_index >= Settings_->StageThree_SubDimension_1 and latent_index < Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2) {

        if (Settings_->StageThree_TrainSubDimension_2 == 0) continue;
        else assert(Settings_->StageThree_TrainSubDimension_2 == 1);

        if (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3) {
          *(LF_C->att(index_C, latent_index)) = SoftThresholdUpdate(thread_id, -1, LF_C->at(index_C, latent_index), LF_C_update);
        }
        else if (Settings_->StageThree_RegularizationType_2 == 2) {
          *(LF_C->att(index_C, latent_index)) -= LF_C_update;
        }
      };

      // *(LF_C->att(index_C, latent_index)) = SoftThresholdUpdate(LF_C->at(index_C, latent_index), LF_C_update);
    }
  }
}

// Compute for S
inline void ThreeMatrixFactorTrainer::ComputeSparse_S_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  Settings  *Settings_     = this->Settings_;

  std::vector<int> indices_B = this->getIndices_B(frame_id, 3);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 3);
  this->ComputeSparse_S_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_B, &indices_C, 1);

  // Load spatial regularization / sliding window indices
  if (Settings_->StageThree_UseSpatialRegularization == 1){
    std::vector<int> indices_BB = this->getIndices_B(frame_id, 31);
    std::vector<int> indices_CC = this->getIndices_C(frame_id, 31);
    this->ComputeSparse_S_Update_Core(thread_id, index_A, frame_id, ground_truth_label, &indices_BB, &indices_CC, 2);
  }
}

inline void ThreeMatrixFactorTrainer::ComputeSparse_S_Update_Core(int thread_id, int index_A, int frame_id, int ground_truth_label, std::vector<int> *indices_B, std::vector<int> *indices_C, int SpatRegLevel) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeSparse_S_Update");

  if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 500000 == 0) {
    PrintFancy(Settings_->session_start_time, "F " + to_string(frame_id) + "ComputeSparseMat_S -- SpatRegLevel " + to_string(SpatRegLevel));
  }

  Settings  *Settings_       = this->Settings_;
  MatrixBlob  *ConditionalScore    = &(this->ConditionalScore);
  Tensor3Blob *Sparse_S        = &(this->Sparse_S);
  Tensor3Blob *Sparse_S_diff     = &(this->Sparse_S_Diff);
  Tensor3Blob *Sparse_S_DiffSq_Cache = &(this->Sparse_S_DiffSq_Cache);

  MatrixBlob  *LF_A          = &(this->LF_A);
  MatrixBlob  *LF_B          = &(this->LF_B);
  MatrixBlob  *LF_C          = &(this->LF_C);

  float Reg_SparseMatS         = 0.;
  float strongweakweight       = 0.;
  float SparseMatS_diff        = 0.;
  float factor             = 0.;

  // Get current score
  float ExponentialScore       = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);

  assert (SpatRegLevel == 1 or SpatRegLevel == 2);
  if (SpatRegLevel == 1){
    Settings_->StageThree_B_SpatReg_Multiplier_Level1;
  } else if (SpatRegLevel == 2){
    Settings_->StageThree_B_SpatReg_Multiplier_Level2;
  }

  // Compute factor for positive / negative label
  if (ground_truth_label == 0) {
    strongweakweight = Settings_->LossWeightWeak;
    factor       = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakweight = Settings_->LossWeightStrong;
    factor       = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  // Check sentinel value!
  if (SpatRegLevel == 1){
    assert(indices_C->size() == 1 + Settings_->DummyMultiplier * Settings_->StageThree_NumberOfNonZeroEntries_C);
  } else if (SpatRegLevel == 2){
    assert( indices_C->size() == 1 + Settings_->DummyMultiplier * (Settings_->StageThree_NumberOfNonZeroEntries_C - Settings_->Dimension_C_BiasSlice) * NUMBER_OF_INTS_WINDOW_3BY3 );
  }

  // Accountancy values
  int index_B_ptr     = 0;
  int index_C_ptr     = 0;
  int index_C_ptr_start = getIndexFirstNonNegative(indices_C);
  float index_B_val   = 0.;
  float index_C_val   = 0.;
  float product_BC    = 0.;
  float product_ABC   = 0.;

  for (int index_B = 0; index_B < Settings_->StageThree_Dimension_B; ++index_B) {

    if (indices_B->at(index_B_ptr) < 0) {
      bh_outofbounds_counter++;
      break;
    }

    if (index_B == indices_B->at(index_B_ptr)) {
      index_B_val = 1.0;
      index_B_ptr += 1;
    } else {
      index_B_val = 0.0;
    }

    index_C_ptr = index_C_ptr_start;

    SKIP_OUT_OF_BOUNDS_INDICES_BH_S3

    // Loop over every defender grid cell
    for (int index_C = 0; index_C < Settings_->StageThree_Dimension_C; ++index_C) {

      if (index_C == indices_C->at(index_C_ptr)) {
        index_C_val = 1.0;
      } else {
        index_C_val = 0.0;
      }

      if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 1000000 == 0 and index_B == 0) {
        cout << setprecision(15) << "frame_id   : " << frame_id << endl;
        cout << setprecision(15) << "index_C_val: " << index_C_val << endl;
        cout << setprecision(15) << "index_C_ptr: " << index_C_ptr << endl;
        cout << setprecision(15) << "index_C  : " << index_C << endl;
      }

      SKIP_OUT_OF_BOUNDS_INDICES_DEF_S3

      if (Settings_->StageThree_UseRepulsion_S_BC == 1) {
        assert (Settings_->StageThree_UseRepulsion_S_ABC == 0);
        product_BC = this->MatrixBlobDotProduct(thread_id, index_B, index_C, LF_B, LF_C);
      }

      if (Settings_->StageThree_UseRepulsion_S_ABC == 1) {
        assert (Settings_->StageThree_UseRepulsion_S_BC == 0);
        product_ABC = this->MatrixBlobDotProductThreeWay(thread_id, index_A, index_B, index_C, LF_A, LF_B, LF_C );
      }

      // old_SparseMatS = Sparse_S->at(index_A, index_B, index_C);

      // SparseMatS_diff = (strongweakweight * factor * index_B_val * index_C_val + Settings_->StageThree_Regularization_RepulsionTerm * product_BC) / Settings_->StageThree_MiniBatchSize;

      SparseMatS_diff = (strongweakweight * factor * index_B_val * index_C_val + Settings_->StageThree_Regularization_RepulsionTerm * product_ABC) / Settings_->StageThree_MiniBatchSize;

      *(Sparse_S_diff->att(index_A, index_B, index_C)) += SparseMatS_diff;

      if (Settings_->EnableDebugPrinter_Level1 == 1 and thread_id == 0 and frame_id % 1000000 == 0 and index_B == 0 and index_C % 20 == 0) { // (isnan(abs(weight_U_diff->at(index_A*n_occ_feat + occ_feat_index)))) {
        PrintFancy(Settings_->session_start_time, "Diagnostics sparse S update");
        cout << setprecision(15) << "Thread ID: " << thread_id << endl;
        cout << setprecision(15) << "BallH ID : " << index_A << endl;
        cout << setprecision(15) << "Frame ID : " << frame_id << endl;
        cout << setprecision(15) << "Multipl  : " << strongweakweight << endl;
        cout << setprecision(15) << "Factor   : " << factor << endl;
        cout << setprecision(15) << "FeatB  : " << index_B << endl;
        cout << setprecision(15) << "FeatB_val: " << index_B_val << endl;
        cout << setprecision(15) << "FeatC  : " << index_C << endl;
        cout << setprecision(15) << "FeatC_val: " << index_C_val << endl;
        cout << setprecision(15) << "RegRepsul: " << Settings_->StageThree_Regularization_RepulsionTerm << endl;
        cout << setprecision(15) << "Prod_ABC : " << product_ABC << endl;
      }


      // If we hit the current index, go to the next one
      if (index_C == indices_C->at(index_C_ptr)){
        index_C_ptr += 1;
      }
      // If we find duplicate index ahead, we keep index_C at its current position
      if (index_C == indices_C->at(index_C_ptr) and index_C_ptr < indices_C->size()-1){
        if (indices_C->at(index_C_ptr + 1) == indices_C->at(index_C_ptr)) {
          index_C -= 1;
        }
      }

    }
  }
}
inline void ThreeMatrixFactorTrainer::ProcessSparse_S_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessSparse_S_Updates");
  Settings  *Settings_       = this->Settings_;
  Tensor3Blob *Sparse_S        = &(this->Sparse_S);
  Tensor3Blob *Sparse_S_diff     = &(this->Sparse_S_Diff);
  Tensor3Blob *Sparse_S_DiffSq_Cache = &(this->Sparse_S_DiffSq_Cache);

  int     n_MiniBatchSize    = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der       = 0.;
  float SparseMatS_update       = 0.;
  float SparseMatS_new        = 0.;
  float adaptive_learning_rate    = 0.;

  adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate;

  // mutex_write_to_Sparse_S.lock();
  for (int index_B = 0; index_B < Settings_->StageThree_Dimension_B; ++index_B) {

    for (int index_C = 0; index_C < Settings_->StageThree_Dimension_C; ++index_C) {

      // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
      // Note that we clamp this number, 1 / sqrt(small number) --> dangerous

      //ADAPTIVE_LEARNING_RATE
      // if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
      //   cum_sum_first_der    = Sparse_S_DiffSq_Cache->at(index_A, index_B, index_C);
      //   adaptive_learning_rate = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);

      //   if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
      //     // PrintFancy(Settings_->session_start_time, "Learning rate update clamped! A: " + to_string(index_A));
      //     adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
      //   }
      // }

      // SparseMatS_update = adaptive_learning_rate * Sparse_S_diff->at(index_A, index_B, index_C);
      // *(Sparse_S->att(index_A, index_B, index_C)) -= SparseMatS_update;

      *(Sparse_S->att(index_A, index_B, index_C)) -= adaptive_learning_rate * Sparse_S_diff->at(index_A, index_B, index_C);

      if (Settings_->StageThree_Sparse_S_UseSoftThreshold == 1) {
        if (Sparse_S->at(index_A, index_B, index_C) < Settings_->StageThree_SparseMatSThreshold) {
          *(Sparse_S->att(index_A, index_B, index_C)) = 0.;
        }
      }
    }
  }
  // cout << "Thr " << thread_id << " BH: " << index_A << " finished updating Sparse mat S" << endl;
}

inline void ThreeMatrixFactorTrainer::Sparse_S_RegularizationUpdate(int thread_id) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeSparse_S_Update");

  Settings  *Settings_ = this->Settings_;
  Tensor3Blob *Sparse_S  = &(this->Sparse_S);

  float learning_rate  = Settings_->StageThree_CurrentLearningRate;
  float Reg_SparseMatS   = Settings_->StageThree_Regularization_Sparse_S_Level1;
  int   n_MiniBatchSize  = Settings_->StageThree_MiniBatchSize;

  float SparseMatS_diff  = 0.;
  float old_SparseMatS   = 0.;
  float sign_SparseMatS  = 0.;

  float grad_at_zero   = 0.;
  float gradient_reg   = 0.;

  std::random_device rd;
  std::mt19937 e2(rd());

  for (int index_A = 0; index_A < Settings_->Dimension_A; ++index_A) {
    for (int index_B = 0; index_B < Settings_->StageThree_Dimension_B; ++index_B) {

      // Regularization sparse matrix S for different levels
      if (index_B == 0)                 Reg_SparseMatS = Settings_->StageThree_Regularization_Sparse_S_Level1;
      if (index_B == Settings_->StageThree_Dimension_B) Reg_SparseMatS = Settings_->StageThree_Regularization_Sparse_S_Level2;

      // Loop over every defender grid cell
      for (int index_C = 0; index_C < Settings_->StageThree_Dimension_C; ++index_C) {

        std::uniform_real_distribution<> dist_b(0, Reg_SparseMatS);

        old_SparseMatS = Sparse_S->at(index_A, index_B, index_C);

        if (Settings_->StageThree_Sparse_S_RegularizationType == 2)
          gradient_reg = old_SparseMatS;
        else if (Settings_->StageThree_Sparse_S_RegularizationType == 1) {
          gradient_reg = this->GetLassoGradient(thread_id, old_SparseMatS);
        }
        else {
          PrintFancy(Settings_->session_start_time, "You didn't specify a valid regularization type! -- Sparse_S");
          return;
        }

        SparseMatS_diff =   Settings_->StageThree_RegularizeS_EveryBatch * \
                  Settings_->StageThree_CurrentLearningRate * \
                  gradient_reg * \
                  Reg_SparseMatS / n_MiniBatchSize;

        // Soft-thresholding: if the gradient takes us past 0, then just leave the weight at 0
        if (Settings_->StageThree_Sparse_S_RegularizationType == 1 and Settings_->StageThree_Sparse_S_UseSoftThreshold == 1) {
          if ( (old_SparseMatS > ZERO_FLOAT and old_SparseMatS - SparseMatS_diff < ZERO_FLOAT) or
             (old_SparseMatS < ZERO_FLOAT and old_SparseMatS - SparseMatS_diff > ZERO_FLOAT)
           ) {
            *(Sparse_S->att(index_A, index_B, index_C)) = ZERO_FLOAT;
          } else {
            *(Sparse_S->att(index_A, index_B, index_C)) -= SparseMatS_diff;
          }
        } else {
          *(Sparse_S->att(index_A, index_B, index_C)) -= SparseMatS_diff;
        }


        if (Settings_->EnableDebugPrinter_Level1 == 1 and index_A == 0 and index_B == 0 and index_C == 0 and thread_id < 5) {
          cout << endl;
          PrintDelimiter(0, 0, 80, '=');
          cout << std::setfill(' ') << setprecision(PRINT_FLOAT_PRECISION) << std::setw(15) << "Regularize S :: " \
             << std::setw(15) << *(Sparse_S->att(index_A, index_B, index_C)) + SparseMatS_diff << " --> " \
             << std::setw(15) << *(Sparse_S->att(index_A, index_B, index_C)) \
             << " || delta: " \
             << std::setw(15) << SparseMatS_diff \
             << " fs: upd_freq " << Settings_->StageThree_RegularizeS_EveryBatch \
             << " lr " << Settings_->StageThree_CurrentLearningRate \
             << " grad " << sign_SparseMatS \
             << " regS " << Reg_SparseMatS \
             << " / minibatchsize " << n_MiniBatchSize \
             << endl;

          PrintDelimiter(0, 0, 80, '=');
          cout << std::setfill(' ');
          cout << endl;
        }

      }
    }
  }
}
inline void ThreeMatrixFactorTrainer::ComputeBias_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeBias_A_Update");

  Settings  *Settings_       = this->Settings_;
  MatrixBlob  *ConditionalScore  = &this->ConditionalScore;
  VectorBlob  *Bias_A        = &this->Bias_A;
  VectorBlob  *Bias_A_Diff     = &this->Bias_A_Diff;
  VectorBlob  *Bias_A_DiffSq_Cache = &this->Bias_A_DiffSq_Cache;

  float ExponentialScore       = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float regularization_Bias_A    = Settings_->StageThree_Regularization_Bias_A;
  int   n_MiniBatchSize      = Settings_->StageThree_MiniBatchSize;

  // Multiplicity in data - there are multiple (#ground_truth_label) mentions of interaction for (A, J)
  float strongweakweight       = 0.;
  float factor           = 0.;

  if (ground_truth_label == 0) {
    strongweakweight = Settings_->LossWeightWeak;
    factor       = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakweight = Settings_->LossWeightStrong;
    factor       = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  float bias                 = Bias_A->data[index_A];
  float bias_diff              = (strongweakweight * factor + regularization_Bias_A * bias) / n_MiniBatchSize;

  *(Bias_A_Diff->att(index_A)) += bias_diff;
  // Bias_A_DiffSq_Cache->at(index_A)  += pow(bias_diff,2);
}
inline void ThreeMatrixFactorTrainer::ProcessBias_A_Updates(int thread_id, int index_A) {
  // // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeWeightW_Update");
  Settings      *Settings_       = this->Settings_;
  VectorBlob      *Bias_A        = &this->Bias_A;
  VectorBlob      *Bias_A_Diff     = &this->Bias_A_Diff;
  VectorBlob      *Bias_A_DiffSq_Cache = &this->Bias_A_DiffSq_Cache;
  int         n_MiniBatchSize    = Settings_->StageThree_MiniBatchSize;

  float cum_sum_first_der          = Bias_A_DiffSq_Cache->at(index_A);

  // Adaptive learning rate        : we divide by the sqrt sum of squared updates of the past.
  // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
  float adaptive_learning_rate       = Settings_->StageThree_CurrentLearningRate;

  // ADAPTIVE_LEARNING_RATE
  if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
    adaptive_learning_rate  = Settings_->StageThree_CurrentLearningRate / sqrt(cum_sum_first_der);
    if (index_A == 0) cout << "Adaptive LearningRate: bias A:" << index_A << " :: " << Settings_->StageThree_CurrentLearningRate << " -> " << adaptive_learning_rate << " using cum_sum_first_der = " << cum_sum_first_der << endl;
    if (adaptive_learning_rate > Settings_->StageThree_Clamp_AdaptiveLearningRate) {
      adaptive_learning_rate = Settings_->StageThree_Clamp_AdaptiveLearningRate;
    }
  }
  *(Bias_A->att(index_A)) -= adaptive_learning_rate * Bias_A_Diff->at(index_A);
}