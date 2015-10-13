#include "core/FullTensorModelTrainer.h"

#include "core/Trainer.h"

using namespace std;
using namespace rapidjson;

using std::chrono::high_resolution_clock;

void FullTensorModelTrainer::Load_SnapshotWeights(string weights_snapshot_file, string momentum_snapshot_file) {
  PrintFancy(Settings_->session_start_time, "Loading weights / biases snapshot " + weights_snapshot_file);
  PrintFancy(Settings_->session_start_time, "Loading momentum     snapshot " + momentum_snapshot_file);

  Settings  *Settings_       = this->Settings_;
  Tensor3Blob *Weight_U        = &this->Weight_U;
  Tensor3Blob *Weight_U_DiffSq_Cache = &this->Weight_U_DiffSq_Cache;
  VectorBlob  *Bias_A        = &this->Bias_A;
  VectorBlob  *Bias_A_DiffSq_Cache   = &this->Bias_A_DiffSq_Cache;
  std::vector<float> *NestorovMomentumLambda = &this->NestorovMomentumLambda;

  string  fn;
  FILE  *fp;
  float   *float_ = new float;

  fn = weights_snapshot_file;
  fp = fopen(fn.c_str() , "rb");

  if (fp == NULL) {
    fputs("File ExponentialScore", stderr);
    exit(1);
  }

  PrintDelimiter(1, 1, 80, '=');

  for (int i = 0; i < Weight_U->data.size(); ++i) {
    fread(float_, sizeof(float), 1, fp);
    Weight_U->data[i] = *float_;
    if (i < 100) {
      cout << "Weight_U[" << i << "] " << *float_ << endl;
    }
  }

  // for (int i = 0; i < Weight_U_DiffSq_Cache->data.size(); ++i) {
  //   fread(float_, sizeof(float), 1, fp);
  //   Weight_U_DiffSq_Cache->data[i] = *float_;
  //   if (i < 100) cout << *float_ << endl;
  // }

  PrintDelimiter(1, 1, 80, '=');

  for (int i = 0; i < Bias_A->data.size(); ++i) {
    fread(float_, sizeof(float), 1, fp);
    Bias_A->data[i] = *float_;
    if (i < 100) {
      cout << "Bias_A[" << i << "] " << *float_ << endl;
    }
  }

  // for (int i = 0; i < Bias_A_DiffSq_Cache->data.size(); ++i) {
  //   fread(float_, sizeof(float), 1, fp);
  //   Bias_A_DiffSq_Cache->data[i] = *float_;
  //   if (i < 100) cout << *float_ << endl;
  // }

  PrintDelimiter(1, 1, 80, '=');

  fclose(fp);

  fn = momentum_snapshot_file;

  // If we didn't specify a momentum snapshot, we just skip it
  if (fn.length() == 0) {
    PrintFancy(Settings_->session_start_time, "No momentum snapshot specified -- skipping loading momentum snapshot");
  } else {

    fp = fopen(fn.c_str() , "rb");

    while (fread(float_, sizeof(float), 1, fp)) {
      NestorovMomentumLambda->push_back(*float_);
    }

    fclose(fp);
  }

  delete float_;

  PrintDelimiter(1, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Resuming training from loaded snapshot");
  PrintDelimiter(1, 1, 80, '=');
}
void FullTensorModelTrainer::InitializeWeightsRandom(float range_W, float range_A) {
  Settings      *Settings_ = this->Settings_;
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist_U(0.0, range_W);
  std::uniform_real_distribution<> dist_b(-range_A, range_A);

  Tensor3Blob *Weight_U = &(this->Weight_U);
  VectorBlob *Bias_A  = &(this->Bias_A);

  for (int i = 0; i < Weight_U->data.size(); ++i) {
    Weight_U->data[i] = dist_U(e2);
  }

  for (int i = 0; i < Bias_A->data.size(); ++i) {
    if (Settings_->StageOne_UseBias == 1) {
      Bias_A->data[i] = dist_b(e2);
    } else {
      Bias_A->data[i] = 0.;
    }
  }
}
void FullTensorModelTrainer::StoreStartingWeights() {
  PrintFancy(Settings_->session_start_time, "Storing weight snapshots");
  Settings *Settings_            = this->Settings_;
  Tensor3Blob *Weight_U          = &this->Weight_U;
  Tensor3Blob *Weight_U_Snapshot = &this->Weight_U_Snapshot;
  VectorBlob  *Bias_A            = &this->Bias_A;
  VectorBlob  *Bias_A_Snapshot   = &this->Bias_A_Snapshot;

  for (int i = 0; i < Weight_U->data.size(); ++i) Weight_U_Snapshot->data[i] = Weight_U->data[i];
  for (int i = 0; i < Bias_A->data.size(); ++i)   Bias_A_Snapshot->data[i] = Bias_A->data[i];
}
void FullTensorModelTrainer::RestoreStartingWeights() {
  PrintFancy(Settings_->session_start_time, "Restoring weight snapshots");
  Settings *Settings_            = this->Settings_;
  Tensor3Blob *Weight_U          = &this->Weight_U;
  Tensor3Blob *Weight_U_Snapshot = &this->Weight_U_Snapshot;
  VectorBlob  *Bias_A            = &this->Bias_A;
  VectorBlob  *Bias_A_Snapshot   = &this->Bias_A_Snapshot;

  for (int i = 0; i < Weight_U->data.size(); ++i)  Weight_U->data[i] = Weight_U_Snapshot->data[i];
  for (int i = 0; i < Bias_A->data.size(); ++i)    Bias_A->data[i] = Bias_A_Snapshot->data[i];
}
int FullTensorModelTrainer::TrainStageOne(string fp_snapshot) {
  // Settings      *Settings_                     = this->Settings_;
  CurrentStateBlob  *CurrentStateBlob          = this->CurrentStateBlob_;
  std::vector<int>  *RandomizedIterator        = &this->RandomizedIterator;
  Tensor3Blob     *Weight_U                    = &this->Weight_U;
  Tensor3Blob     *Weight_U_Diff               = &this->Weight_U_Diff;
  VectorBlob      *Bias_A                      = &this->Bias_A;
  VectorBlob      *Bias_A_Diff                 = &this->Bias_A_Diff;
  std::vector<float>  *NestorovMomentumLambda_ = &this->NestorovMomentumLambda;

  // Debug mode?
  if (Settings_->EnableDebugPrinter_Level1 == 1) {
    // EnableDebugMode();
    // spatial_entropy.EnableDebugMode();
    spatial_entropy_sign.EnableDebugMode();
  } else {
    DisableDebugMode();
    spatial_entropy.DisableDebugMode();
    spatial_entropy_sign.DisableDebugMode();
  }

  string logfile_prefix = to_string(CurrentStateBlob_->session_id);
  string logfile_suffix = "_full" + to_string(Settings_->StageOne_Dimension_B);
  initLogFiles(logfile_prefix, logfile_suffix);

  // Set the correct number of threads to use in generateTrainingBatches -- this is a slight hack
  Settings_->CurrentNumberOfThreads = Settings_->StageOne_NumberOfThreads;
  CurrentStateBlob->current_stage = 1;

  int *have_recorded_a_trainloss = new int;
  *have_recorded_a_trainloss = 0;

  if (Settings_->EnableDebugPrinter_Level1 == 1){
    PrintWithDelimiters(Settings_->session_start_time, "Debug print ENABLED --> showing peeks into internal numbers");
  } else {
    PrintWithDelimiters(Settings_->session_start_time, "Debug print DISABLED");
  }

  // Initialize the parameters to be learned
  if (Settings_->ResumeTraining == 1) {
    this->Load_SnapshotWeights(Settings_->StageOne_SnapshotWeightsBiases, Settings_->StageOne_SnapshotMomentum);
  } else {
    this->InitializeWeightsRandom(
      Settings_->StageOne_Initialization_Weight_U_range,
      Settings_->StageOne_Initialization_Bias_A_range
      );
  }

  PrintFancy(Settings_->session_start_time, "Debug -- Bias_A initialization peek");
  Bias_A->showVectorContents(100, 10);

  PrintFancy(Settings_->session_start_time, "Debug -- Weight_U initialization peek");
  Weight_U->showTensorContents(10, 1, 10, 1, Settings_->StageOne_Dimension_C - 1);

  // Start worker threads
  std::vector<boost::thread *> threads(Settings_->StageOne_NumberOfThreads);
  for (int thread_id = 0; thread_id < Settings_->StageOne_NumberOfThreads; ++thread_id) {
    threads[thread_id] = new boost::thread(&FullTensorModelTrainer::ThreadComputer, this, thread_id);
  }
  int n_total_labels_train  = Settings_->NumberOfWeakLabels_Train + Settings_->NumberOfStrongLabels_Train;

  // ------------------------------------------------------------------------------------------------------------------------------
  // Xval runs
  // ------------------------------------------------------------------------------------------------------------------------------
  for (int cross_val_run = Settings_->StageOne_StartFromCrossValRun; cross_val_run < Settings_->StageOne_NumberOfCrossValidationRuns; ++cross_val_run) {

    PrintFancy(Settings_->session_start_time, "M | Starting cross-validation run: " + to_string(cross_val_run));
    PrintFancy(Settings_->session_start_time, "M | ================================\n");
    CurrentStateBlob->current_cross_val_run = cross_val_run;

    // We do cross-validation, so every x-validation run, we choose a different 10% (the next 10% in the train+val set) as the validation set
    this->PermuteTrainValIndices();

    // A snapshot of the starting weights, to be restored if the scores blow up
    this->StoreStartingWeights();

    *have_recorded_a_trainloss = 0;

    // We also want to go randomly through the chosen training-set -- shuffle the train-indices
    // Reset learning learning_rate
    Settings_->StageOne_CurrentLearningRate                         = Settings_->StageOne_StartingLearningRate;
    Settings_->StageOne_Cumulative_sum_of_squared_first_derivatives = 0.;
    Settings_->StageOne_LossTrainPreviousEpoch                      = LOSS_INIT_VALUE;
    Settings_->StageOne_LossValidPreviousEpoch                      = LOSS_INIT_VALUE;

    int epoch = Settings_->StageOne_StartFromEpoch;

    // ------------------------------------------------------------------------------------------------------------------------------
    // Epochs: 1 epoch == 1 pass over dataset
    // ------------------------------------------------------------------------------------------------------------------------------
    while (epoch < Settings_->StageOne_NumberOfEpochs) {

      PrintFancy(Settings_->session_start_time, "M | Starting epoch: " + to_string(epoch));
      PrintWithDelimiters("Current learning-rate: " + to_string(Settings_->StageOne_CurrentLearningRate));

      high_resolution_clock::time_point start_time_epoch = high_resolution_clock::now();
      CurrentStateBlob->current_epoch = epoch;

      int n_batches_served                         = 0;
      int n_batches                                = static_cast<int>(floor(static_cast<float>(n_total_labels_train) / (Settings_->StageOne_NumberOfThreads * Settings_->StageOne_MiniBatchSize) )) + 1;
      int batch_size                               = Settings_->StageOne_NumberOfThreads;
      int last_batch_size                          = 0;
      int n_threads_commanded_this_batch           = 0;
      int n_datapoints_processed_so_far_this_epoch = 0;

      // Iterator logic:
      // There are gtruth labels, which we partitioned in train-valid-test
      // We shuffled train+valid at the start of xval-runs, and permute the 10% validation through the data-set

      // We then need to go through the train-part randomly!
      // There are two indices.
      // 1. A global index that keeps track of how many training examples we have seen.
      // 2. An index that chooses which element in the train-set to present next.

      // Shuffle the iterators - we do this once every epoch / each run through the entire data-set.
      mt19937 g(static_cast<uint32_t>(time(0)));
      std::shuffle(RandomizedIterator->begin(), RandomizedIterator->end(), g);
      std::shuffle(RandomizedIterator->begin(), RandomizedIterator->end(), g);

      this->PrintStatus();

      if (epoch % Settings_->StageOne_ResetMomentumEveryNthEpoch == 0) {
        PrintFancy(Settings_->session_start_time, "Resetting Nestorov momentum");
        NestorovMomentumLambda_->resize(0);
      }

      PrintFancy(Settings_->session_start_time, "M | Computing terms for train-set. Looping over all strong+weak train labels.");


      // ------------------------------------------------------------------------------------------------------------------------------
      // Clear all empirical frequencies of gradients.
      // ------------------------------------------------------------------------------------------------------------------------------
      PrintFancy() << "Erasing counts of gradients.";
      spatial_entropy.EraseHistograms();
      spatial_entropy_sign.EraseHistograms();

      // ------------------------------------------------------------------------------------------------------------------------------
      // Minibatch stochastic gradient descent
      // ------------------------------------------------------------------------------------------------------------------------------

      for (int batch = 0; batch < n_batches; batch++) {

        // ------------------------------------------------------------------------------------------------------------------------------
        // Clear all empirical frequencies of gradients.
        // ------------------------------------------------------------------------------------------------------------------------------
        if (Settings_->EraseEntropyEveryBatch == 1) {
          // PrintFancy() << "Erasing counts of gradients @ batch " << batch << endl;
          // spatial_entropy.EraseHistograms();
          // spatial_entropy_sign.EraseHistograms();
        }

        if (n_batches_served % Settings_->StageOne_StatusEveryNBatchesTrain == 0) {
          PrintFancy(Settings_->session_start_time, "M | Batch "+to_string(n_batches_served) );
          this->DebuggingTestProbe(0, 0, 0, 0, 0);
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Forward-prop: ComputeConditionalScore
        // ------------------------------------------------------------------------------------------------------------------------------
        // if (batch % Settings_->StageOne_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeConditionalScore");
        n_threads_commanded_this_batch = this->generateTrainingBatches(1, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        if (global_reset_training == 1) {

          // For the current xval-run
          Settings_->StageOne_CurrentLearningRate   *= SCORE_BLOWUP_ADJUST_LR_BY_FACTOR;

          // For the next xval-run
          Settings_->StageOne_StartingLearningRate  *= SCORE_BLOWUP_ADJUST_LR_BY_FACTOR;

          PrintWithDelimiters("Score blowup detected: Adjusting LR " + to_string(Settings_->StageOne_CurrentLearningRate / SCORE_BLOWUP_ADJUST_LR_BY_FACTOR) + " --> " + to_string(Settings_->StageOne_CurrentLearningRate));

          // Reset all weights to before the diffs were applied this batch
          this->RestoreStartingWeights();

          // Break out of the loop over the data-set
          break;
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Backprop: Compute gradient
        // ------------------------------------------------------------------------------------------------------------------------------
        // Compute change in weight matrix W
        // if (batch % Settings_->StageOne_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ComputeWeight_U_Update");
        n_threads_commanded_this_batch = this->generateTrainingBatches(2, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        // Compute change in bias for actions
        if (Settings_->StageOne_UseBias == 1) {
          n_threads_commanded_this_batch = this->generateTrainingBatches(4, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Backprop: Apply computed updates to weight matrix W
        // ------------------------------------------------------------------------------------------------------------------------------
        n_threads_commanded_this_batch = this->generateTrainingBatches(3, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        // Apply computed updates to Bias_A
        if (Settings_->StageOne_UseBias == 1) {
          // if (batch % Settings_->StageOne_StatusEveryNBatchesTrain == 0) PrintFancy(Settings_->session_start_time, to_string(batch) + " | starting ProcessBias_A_Updates");
          n_threads_commanded_this_batch = this->generateTrainingBatches(5, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Compute spatial entropy of gradients
        // ------------------------------------------------------------------------------------------------------------------------------
        // Add gradients of minibatch to histogram -- this is paralellized
        n_threads_commanded_this_batch = this->generateTrainingBatches(9, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
        task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

        // Every X batches, we compute and log the spatial entropy.
        if (batch > 0 and debug_mode and batch % 100 == 0) {
          spatial_entropy.ShowEntropies();
        }
        if (batch > 0 and batch % Settings_->ComputeEntropyFrequency == 0) {
          high_resolution_clock::time_point start_time_entropy = high_resolution_clock::now();
          // Compute and log entropy.
          ComputeEntropy();
          // LogEntropiesToFile(); // entropy over time
          LogEntropy(); // average entropy
          // spatial_entropy.ShowEntropies();

          if (Settings_->EraseEntropyEveryBatch == 1) {
            spatial_entropy.EraseHistograms();
            spatial_entropy_sign.EraseHistograms();
          }

          PrintTimeElapsedSince(start_time_entropy, "Entropy compute time: ");
        }
        if (batch > 0 and batch % Settings_->ComputeTrainLossFrequency == 0) {
          float loss_train = ComputeLoss(6);
          // Compute and log train loss.
          IOController_->WriteToFile(logfile_loss_filename_, to_string(GetTimeElapsedSince(Settings_->session_start_time)) + ",");
          IOController_->WriteToFile(logfile_loss_filename_, to_string(loss_train) + ",");
          IOController_->WriteToFile(logfile_loss_filename_, to_string('-1') + "\n");
          PrintFancy() << "Loss (train): " << loss_train << endl;
        }
        if (batch > 0 and batch % Settings_->WriteProbabilitiesFrequency == 0) {
          LogProbabilitiesToFile();
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Apply momentum
        // ------------------------------------------------------------------------------------------------------------------------------
        if (batch % Settings_->StageOne_ApplyMomentumEveryNthMinibatch == 0) {
          this->ApplyNestorovMomentum(batch);
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Threshold weights
        // ------------------------------------------------------------------------------------------------------------------------------
        if (Settings_->StageOne_Weight_U_ClampToThreshold == 1 and batch % 5 == 0) {
          Weight_U->ThresholdValues(-1e99, Settings_->StageOne_Weight_U_Threshold, Settings_->StageOne_Weight_U_Threshold);
        }

        // ------------------------------------------------------------------------------------------------------------------------------
        // Debug -- peek at weights
        if (Settings_->EnableDebugPrinter_Level3 == 1) {
          cout << "Debug -- Weight_U peek -- batch " << batch << endl;
          Weight_U->showTensorContents(10, 1, 10, 1, 36);
          cout << "Debug -- Weight_U_Diff peek -- batch " << batch << endl;
          Weight_U_Diff->showTensorContents(10, 1, 10, 1, 36);
        }
        // ------------------------------------------------------------------------------------------------------------------------------

        if (Settings_->StageOne_Weight_U_ClampToThreshold == 1){
          if (Settings_->EnableDebugPrinter_Level2 == 1) {
            cout << "Thresholding Weight_U to " << Settings_->StageOne_Weight_U_Threshold << endl;
          }
          Weight_U->ThresholdValues(-1e99, Settings_->StageOne_Weight_U_Threshold, Settings_->StageOne_Weight_U_Threshold);
        }

        // Clear the differences computed
        Weight_U_Diff->erase();
        Bias_A_Diff->erase();

        // Note this miscounts at the last batch (n_batches - 1), because that batch might contain a smaller number of datapoints than MiniBatchSize
        // but this is irrelevant there anyway [EXCEPT FOR MISUNDERSTANDINGS DURING DEBUGGING]
        n_datapoints_processed_so_far_this_epoch += n_threads_commanded_this_batch * Settings_->StageOne_MiniBatchSize;
        n_batches_served++;
      }

      PrintTimeElapsedSince(start_time_epoch, "Batch train-time: ");

      // If a blow-up was detected, weights were reset and we skip the loss computations
      if (global_reset_training == 1) {
        // Reset the epoch counter (but we stay in the current xval-run)
        epoch = Settings_->StageOne_StartFromEpoch;
        global_reset_training = 0;
        continue;
      }

      // Compute losses now
      if (epoch % Settings_->StageOne_ComputeLossAfterEveryNthEpoch == 0) {

        high_resolution_clock::time_point start_time_ComputeValidTestSetScores = high_resolution_clock::now();

        PrintFancy(Settings_->session_start_time, "ComputeConditionalScore Validation");
        // TODO(Stephan) MT - We need to recalculate the ExponentialScores (for train, only ExponentialScores for train-set have been done)!
        int n_total_labels_valid = Settings_->NumberOfWeakLabels_Val + Settings_->NumberOfStrongLabels_Val;
        n_batches_served         = 0;
        n_batches                = static_cast<int>(floor(static_cast<float>(n_total_labels_valid) / (Settings_->StageOne_NumberOfThreads * Settings_->StageOne_MiniBatchSize) )) + 1;
        batch_size               = Settings_->StageOne_NumberOfThreads;
        last_batch_size          = 0;
        n_datapoints_processed_so_far_this_epoch = 0;

        for (int batch = 0; batch < n_batches; batch++) {
          if (n_batches_served % Settings_->StageOne_StatusEveryNBatchesValid == 0) PrintFancy(Settings_->session_start_time, "M | Batch "+to_string(n_batches_served) );

          // ComputeConditionalScore
          n_threads_commanded_this_batch = this->generateValidationBatches(1, batch, n_batches, n_datapoints_processed_so_far_this_epoch, Settings_->StageOne_MiniBatchSize);
          task_queue_.waitForTasksToComplete(n_threads_commanded_this_batch);

          // Note this miscounts at the last batch (n_batches - 1), because that batch might contain a smaller number of datapoints than MiniBatchSize
          // but this is irrelevant there anyway [EXCEPT FOR MISUNDERSTANDINGS DURING DEBUGGING]
          n_datapoints_processed_so_far_this_epoch += n_threads_commanded_this_batch * Settings_->StageOne_MiniBatchSize;

          n_batches_served++;
        }

        // this->ComputeConditionalScoreSingleThreaded(7);

        PrintFancy(Settings_->session_start_time, "ComputeConditionalScore Test");
        ComputeConditionalScoreSingleThreaded(8);
        PrintTimeElapsedSince(start_time_ComputeValidTestSetScores, "Valid / test ExponentialScore compute-time: ");

        // All ExponentialScores have been computed, so we can compute the loss

        high_resolution_clock::time_point start_time_loss_compute = high_resolution_clock::now();
        float loss_train = this->ComputeLoss(6);
        float loss_valid = this->ComputeLoss(7);
        float loss_test  = LOSS_INIT_VALUE; //this->ComputeLoss(8);
        this->ProcessComputedLosses(cross_val_run, epoch, loss_train, loss_valid, loss_test);

        PrintFancy(Settings_->session_start_time, "Epoch results -- xval-run: "+to_string(cross_val_run)+" | epoch "+to_string(epoch));

        cout << ">>>> Train loss: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_train;
        cout << " -- prev: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << Settings_->StageOne_LossTrainPreviousEpoch;
        cout << " -- delta: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_train - Settings_->StageOne_LossTrainPreviousEpoch;
        cout << endl;

        cout << ">>>> Valid loss: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_valid;
        cout << " -- prev: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << Settings_->StageOne_LossValidPreviousEpoch;
        cout << " -- delta: " << setprecision(PRINT_FLOAT_PRECISION_LOSS) << loss_valid - Settings_->StageOne_LossValidPreviousEpoch;
        cout << endl;

        PrintTimeElapsedSince(start_time_loss_compute, "Loss compute-time");

        // =================================================================================================
        // Adjust learning-rates and decide whether to continue to the next epoch / xval-run, or end the run
        // =================================================================================================

        if (this->DecisionUnit(cross_val_run, epoch, loss_train, loss_valid, loss_test, have_recorded_a_trainloss) == 0) break;

        // -------------------------------------------------------------------------------------------------
        // Write losses to log-file
        // -------------------------------------------------------------------------------------------------
        IOController_->WriteToFile(logfile_loss_filename_, to_string(GetTimeElapsedSince(Settings_->session_start_time)) + ",");
        IOController_->WriteToFile(logfile_loss_filename_, to_string(loss_train) + ",");
        IOController_->WriteToFile(logfile_loss_filename_, to_string(loss_valid) + "\n");
        // -------------------------------------------------------------------------------------------------
      }

      // Take a snapshot - store the learned parameters
      if (epoch % Settings_->StageOne_Take_SnapshotAfterEveryNthEpoch == 0 || epoch == Settings_->StageOne_NumberOfEpochs - 1) {
        PrintFancy(Settings_->session_start_time, "M | Taking snapshot | Exporting learned parameters to directory -- " + fp_snapshot);
        this->Store_Snapshot(cross_val_run, epoch, fp_snapshot);
      }

      // Show Entropy summary.
      spatial_entropy.ShowSummary();
      // ------------------------------------------------------------------------------------------------
      // End of epoch
      // ------------------------------------------------------------------------------------------------
      // Loss did not below threshold, but we hit the max #epochs
      if (epoch == Settings_->StageOne_NumberOfEpochs - 1) PrintFancy(Settings_->session_start_time, "Maximal number of epochs reached, but not all loss thresholds reached... continuing to next xval-run.");

      epoch++;
    }
  }

  // Send worker threads the KILL signal
  for (int i = 0; i < Settings_->StageOne_NumberOfThreads; ++i) {
    QueueMessage QueueMessageSend_;
    QueueMessageSend_.task_type = 0;
    QueueMessageSend_.index_A = _dummy_index_A_to_test_mt+i;

    // TODO(Stephan) Change this: getTask() routine should read 1 at a time?
    task_queue_.mutex_.lock();
    for (int i = 0; i < Settings_->StageOne_MiniBatchSize; ++i) {
      task_queue_.taskQueue.push(QueueMessageSend_);
    }
    task_queue_.mutex_.unlock();
    task_queue_.qGoFetch.notify_one();
  }

  for (int thread_id = 0; thread_id < Settings_->StageOne_NumberOfThreads; ++thread_id) {
    if (threads[thread_id]->joinable()) {
      PrintFancy(Settings_->session_start_time, "Waiting for thread " + to_string(thread_id));
      threads[thread_id]->join();
    }
  }
  for (int thread_id = 0; thread_id < Settings_->StageOne_NumberOfThreads; ++thread_id) {
    delete threads[thread_id];
  }

  delete have_recorded_a_trainloss;

  PrintFancy(Settings_->session_start_time, "Workers are all done -- end of Stage 3.");
  return 0;
}
int FullTensorModelTrainer::DecisionUnit(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test, int *have_recorded_a_trainloss){

  // Return 1 if training has to continue, 0 if training has to end.

  Settings *Settings_ = this->Settings_;

  // abs(loss_train) < settings->StageOne_TrainingConditionalScoreThreshold \
  // || abs(loss_valid) < settings->StageOne_ValidationConditionalScoreThreshold \
  // || abs(loss_test) < settings->StageOne_TestConditionalScoreThreshold

  // If validation loss goes up again -- not implemented here
  // if ( loss_valid <= 0 ) {
  //   // Do stuff here
  //   PrintFancy(Settings_->session_start_time, "M | Stage 3: validation loss delta thresholds reached");
  //   PrintFancy(Settings_->session_start_time, "M | Exporting learned parameters to file -- this->Store_Snapshot");
  //   this->Store_Snapshot(cross_val_run, epoch, fp_snapshot);
  //   break;
  // }

  PrintDelimiter(1, 0, 80, '=');
  PrintFancy(Settings_->session_start_time, "Deciding what the next step is -- did we record a train-loss before? >> " + to_string(*have_recorded_a_trainloss));
  PrintDelimiter(0, 1, 80, '=');

  if (*have_recorded_a_trainloss == 1) {

    // Train / valid loss blew up?
    if (CheckIfFloatIsNan(loss_train, "") \
      || CheckIfFloatIsNan(loss_valid, "") \
      || abs(loss_train) > LOSS_BLOWUP_THRESHOLD \
      || abs(loss_valid) > LOSS_BLOWUP_THRESHOLD) {

      PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " blew up!");
      PrintFancy(Settings_->session_start_time, "Adjusting learning-rate + resetting weights ");

      cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;

      PrintFancy(Settings_->session_start_time, "Will try a new epoch");

      Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;
      this->RestoreStartingWeights();

    } // Validation loss went up? --> Drop LR by a lot OR quit training
    else if (loss_valid - Settings_->StageOne_LossValidPreviousEpoch > 0 and abs(loss_valid - Settings_->StageOne_LossValidPreviousEpoch) > Settings_->StageOne_ValidationLossIncreaseThreshold) {

      PrintDelimiter(1, 1, 80, '=');
      PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " > " +to_string(Settings_->StageOne_LossValidPreviousEpoch));

      // PrintFancy(Settings_->session_start_time, "Adjusting learning-rate: ");
      // cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY << endl;
      // Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY;

      PrintFancy(Settings_->session_start_time, "Validation delta over threshold, quitting this xval-run and going to next xval-run.");
      PrintDelimiter(1, 1, 80, '=');
      return 0;

    } // Validation loss decreased too slowly? --> Drop LR by a lot OR quit training
    else if (loss_valid - Settings_->StageOne_LossValidPreviousEpoch < 0 and abs(loss_valid - Settings_->StageOne_LossValidPreviousEpoch) < Settings_->StageOne_ValidationLossDecreaseThreshold) {

      PrintDelimiter(1, 1, 80, '=');
      PrintFancy(Settings_->session_start_time, "ValidLoss now: " + to_string(loss_valid) + " < " +to_string(Settings_->StageOne_LossValidPreviousEpoch));

      // PrintFancy(Settings_->session_start_time, "Adjusting learning-rate: ");
      // cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY << endl;
      // Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY;

      PrintFancy(Settings_->session_start_time, "Validation delta under threshold, quitting this xval-run and going to next xval-run.");
      PrintDelimiter(1, 1, 80, '=');
      return 0;

    } else {
      // Validation loss did not go up, what did train-loss do?
      if (loss_train > Settings_->StageOne_LossTrainPreviousEpoch) {

        PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " > " +to_string(Settings_->StageOne_LossTrainPreviousEpoch));
        PrintFancy(Settings_->session_start_time, "Decreasing learning-rate: ");
        PrintDelimiter(1, 1, 80, '=');
        cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;
        PrintDelimiter(1, 1, 80, '=');
        Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;

      } else {

        PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " <= " +to_string(Settings_->StageOne_LossTrainPreviousEpoch));
        PrintFancy(Settings_->session_start_time, "Increasing learning-rate: ");
        PrintDelimiter(1, 1, 80, '=');
        cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_DOWN_CHANGE_RATE_BY << endl;
        PrintDelimiter(1, 1, 80, '=');
        Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_DOWN_CHANGE_RATE_BY;
        // Store the better (= lower) training loss and go to next epoch
        Settings_->StageOne_LossTrainPreviousEpoch = loss_train;
        Settings_->StageOne_LossValidPreviousEpoch = loss_valid;

      }
    }

  } else {

    if (CheckIfFloatIsNan(loss_train, "") ||
        CheckIfFloatIsNan(loss_valid, "") ||
        abs(loss_train) > LOSS_BLOWUP_THRESHOLD ||
        abs(loss_valid) > LOSS_BLOWUP_THRESHOLD) {

      // Train / valid loss blew up?

      PrintFancy(Settings_->session_start_time, "TrainLoss now: " + to_string(loss_train) + " blew up!");
      PrintFancy(Settings_->session_start_time, "Adjusting learning-rate + resetting weights ");
      cout << setprecision(PRINT_FLOAT_PRECISION_LR) << Settings_->StageOne_CurrentLearningRate << " --> " << Settings_->StageOne_CurrentLearningRate * LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY << endl;

      PrintFancy(Settings_->session_start_time, "Will try again with lower LR");
      Settings_->StageOne_CurrentLearningRate *= LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY;

    } else {

      PrintFancy(Settings_->session_start_time, "No previous train / valid loss recorded, and losses did not blow up. Storing and going to next epoch.");
      Settings_->StageOne_LossTrainPreviousEpoch = loss_train;
      Settings_->StageOne_LossValidPreviousEpoch = loss_valid;
      *have_recorded_a_trainloss = 1;

    }

  }

  return 1;
}
void FullTensorModelTrainer::ThreadComputer(int thread_id) {
  Settings *Settings_ = this->Settings_;
  int MiniBatchSize   = Settings_->StageOne_MiniBatchSize;

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
      if (task.task_type == 2) {
        // PrintFancy(Settings_->session_start_time, "ComputeWeight_U_Update");
        this->ComputeWeight_U_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 3) {
        // PrintFancy(Settings_->session_start_time, "ProcessWeight_U_Updates");
        this->ProcessWeight_U_Updates(thread_id, task.index_A);
      }
      if (task.task_type == 4) {
        // PrintFancy(Settings_->session_start_time, "ComputeBias_A_Update");
        this->ComputeBias_A_Update(thread_id, task.index_A, task.frame_id, task.ground_truth_label);
      }
      if (task.task_type == 5) {
        // PrintFancy(Settings_->session_start_time, "ProcessBias_A_Updates");
        this->ProcessBias_A_Updates(thread_id, task.index_A);
      }

      // 6 = train-loss
      // 7 = valid-loss
      // 8 = test -loss
      if (task.task_type == 6 || task.task_type == 7 || task.task_type == 8) {
        // PrintFancy(Settings_->session_start_time, "ComputeLossPartial");
        // this->ComputeLossPartial(thread_id, task.labels_id_start, task.labels_id_end, task.index_A_start, task.index_A_end, task.task_type)
      }

      if (task.task_type == 9) {
        this->AddGradientsToHistograms(thread_id, task.index_A, task.frame_id, task.ground_truth_label,
          Settings_->StageOne_Dimension_B,
          Settings_->StageOne_Dimension_C,
          1.0,
          Settings_->StageOne_MiniBatchSize);
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
void FullTensorModelTrainer::DebuggingTestProbe(int thread_id, int index_A, int frame_id, int ground_truth_label, int index) {
  // Settings     *Settings_   = this->Settings_;
  // MatrixBlob *Weight_U = &(this->Weight_U);
  // MatrixBlob *OccupancyFeat = this->Blob_B_;
  // VectorBlob *Bias_A    = &(this->Bias_A);

  // PrintFancy(Settings_->session_start_time, "Running DebuggingTestProbe");
  // for (int index = 0; index < 8192; ++index) {
  //   if (index % 500 == 0) {
  //     cout << "What's in U? " << index_A << " " << index << " " << Weight_U->data[Weight_U->SerialIndex(index_A, index)] << endl;
  //   }
  //   if (abs(Weight_U->data[Weight_U->SerialIndex(index_A, index)]) > 0.001) {
  //     cout << ">>> Big value spotted: " << index_A << " " << index << " " << Weight_U->data[Weight_U->SerialIndex(index_A, index)] << endl;
  //   }
  // }
}
void FullTensorModelTrainer::ApplyNestorovMomentum(int batch) {
  // PrintFancy(Settings_->session_start_time, "Applying momentum");
  Settings      *Settings_                             = this->Settings_;
  std::vector<float>  *NestorovMomentumLambda_         = &this->NestorovMomentumLambda;
  std::vector<float>  *NestorovMomentumPreviousWeight_ = &this->NestorovMomentumPreviousWeight;
  std::vector<float>  *NestorovMomentumPreviousBias_A_ = &this->NestorovMomentumPreviousBias_A;

  Tensor3Blob *Weight_U                                = &(this->Weight_U);
  Tensor3Blob *Weight_U_diff                           = &(this->Weight_U_Diff);
  Tensor3Blob *Weight_U_DiffSq_Cache                   = &(this->Weight_U_DiffSq_Cache);

  VectorBlob  *Bias_A                                  = &this->Bias_A;
  VectorBlob  *Bias_A_Diff                             = &this->Bias_A_Diff;
  VectorBlob  *Bias_A_DiffSq_Cache                     = &this->Bias_A_DiffSq_Cache;

  int n_dimension_A                                    = Settings_->Dimension_A;
  int n_dimension_B                                    = Settings_->StageOne_Dimension_B;
  int n_dimension_C                                    = Settings_->StageOne_Dimension_C;

  float lambda                                         = 0.;
  float lambda_new                                     = 0.;
  float gamma                                          = 0.;

  // For conventions on old and new lambdas / gammas, see
  // https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
  // PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) + " | doing momentum updates");
  int   n_momentum_updates_seen   = NestorovMomentumLambda_->size();

  if (n_momentum_updates_seen == 0) {

    PrintFancy(Settings_->session_start_time, "No previous momentum updates, initializing now.");
    NestorovMomentumLambda_->push_back(1.);

    for (int i = 0; i < n_dimension_B * n_dimension_C * n_dimension_A; ++i)
    {
      NestorovMomentumPreviousWeight_->push_back(Weight_U->data[i]);
    }

    if (Settings_->StageOne_UseBias == 1) {
      for (int i = 0; i < n_dimension_A; ++i) {
        NestorovMomentumPreviousBias_A_->push_back(Bias_A->data[i]);  // Store the first weight snapshot for the momentum updates
      }
    }

  } else {  // Apply the momentum update

    // Momentum factor
    lambda   = NestorovMomentumLambda_->at(n_momentum_updates_seen - 1);
    lambda_new = 0.5 * (1 + sqrt(1 + 4 * pow(lambda, 2) ) );

    NestorovMomentumLambda_->push_back(lambda_new);

    this->BoostParameter(batch, Weight_U, Weight_U_diff, Weight_U_DiffSq_Cache, NestorovMomentumPreviousWeight_, n_dimension_A * n_dimension_C * n_dimension_B );

    if (Settings_->StageOne_UseBias == 1) {
      this->BoostParameter(batch, Bias_A, Bias_A_Diff, Bias_A_DiffSq_Cache, NestorovMomentumPreviousBias_A_, n_dimension_A);
    }
  }
}
float FullTensorModelTrainer::ComputeTripleProductPsiPsiU(int thread_id, int frame_id, int index_A, Tensor3Blob *Weight_U, std::vector<int> *indices_B, std::vector<int> *indices_C ) {
  float sum = 0.;

  for (int index_B = 0; index_B < indices_B->size() - 1; ++index_B)
  {

    SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S1
    if (indices_B->at(index_B) < 0) continue;

    for (int index_C = 0; index_C < indices_C->size() - 1; ++index_C)
    {
      SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S1
      if (indices_C->at(index_C) < 0) continue;

      sum += Weight_U->at(index_A, indices_B->at(index_B), indices_C->at(index_C));
      // Debug!
      // if (frame_id % 1000000 == 0 and index_C == 0) {
      //   cout << "indices_B->at(index_B) " << indices_B->at(index_B) << endl;
      //   cout << "indices_C->at(index_C)   " << indices_C->at(index_C) << endl;
      //   cout << "sum                  " << sum << endl;
      // }
    }
  }
  return sum;
}
float FullTensorModelTrainer::ComputeScore(int thread_id, int index_A, int frame_id) {
  Settings    *Settings_ = this->Settings_;
  Tensor3Blob *Weight_U  = &(this->Weight_U);
  VectorBlob  *Bias_A    = &(this->Bias_A);

  float     score        = 0.;

  std::vector<int> indices_B = this->getIndices_B(frame_id, 1);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 1);

  score += this->ComputeTripleProductPsiPsiU( thread_id,
                        frame_id,
                        index_A,
                        Weight_U,
                        &indices_B,
                        &indices_C
                        );

  // Add the bias
  score += Bias_A->at(index_A);

  return score;
}
int FullTensorModelTrainer::ComputeConditionalScore(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeConditionalScore");

  Settings  *Settings_          = this->Settings_;
  MatrixBlob  *ConditionalScore = &(this->ConditionalScore);

  float score = this->ComputeScore(thread_id, index_A, frame_id);

  if (CheckIfFloatIsNan(score, "") || score > SCORE_BLOWUP_THRESHOLD_ABOVE || score < SCORE_BLOWUP_THRESHOLD_BELOW) {
    PrintFancy(Settings_->session_start_time, "Blowup! Task: " + to_string(index_A) + " f: " + to_string(frame_id) + " g: " + to_string(ground_truth_label) + " s: " + to_string(score));
    global_reset_training = 1;
    return -1;
  }

  *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)) = score;

  if (Settings_->EnableDebugPrinter_Level1 == 1){
    if (frame_id % 100000 == 0) {
      cout << "BallHandler ID : " << index_A << endl;
      cout << "Frame ID       : " << frame_id << endl;
      cout << "Action         : " << ground_truth_label << endl;
      cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << "Score      : " << score << endl;
    }
  }

  return 0;
}
void FullTensorModelTrainer::ComputeWeight_U_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeWeight_U_Update");

  Settings  *Settings_               = this->Settings_;
  MatrixBlob  *ConditionalScore      = &(this->ConditionalScore);
  Tensor3Blob *Weight_U              = &(this->Weight_U);
  Tensor3Blob *Weight_U_diff         = &(this->Weight_U_Diff);
  Tensor3Blob *Weight_U_DiffSq_Cache = &(this->Weight_U_DiffSq_Cache);

  int n_dimension_B                  = Settings_->StageOne_Dimension_B;
  int n_dimension_C                  = Settings_->StageOne_Dimension_C;

  float ExponentialScore             = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float learning_rate                = Settings_->StageOne_CurrentLearningRate;
  int   n_MiniBatchSize              = Settings_->StageOne_MiniBatchSize;
  float regularizationU              = Settings_->StageOne_Regularization_Weight_U_Level1;

  float strongweakweight             = 0.;
  float factor                       = 0.;
  float U_gradient                   = 0.;
  float index_B_val                  = 0.;
  float index_C_val                  = 0.;
  float old_weight                   = 0.;
  float gradient_reg                 = 0.;
  float grad_at_zero                 = 0.;
  int serial_index                   = 0;

  if (ground_truth_label == 0) {
    strongweakweight = Settings_->LossWeightWeak;
    factor       = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakweight = Settings_->LossWeightStrong;
    factor       = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  std::vector<int> indices_B = this->getIndices_B(frame_id, 1);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 1);
  assert(indices_C.size() == 1 + Settings_->DummyMultiplier * (Settings_->StageOne_NumberOfNonZeroEntries_C));

  if (Settings_->EnableDebugPrinter_Level2 == 1 and frame_id % 10000 == 0) {
    cout << "T: " << thread_id << " Frame " << frame_id << " indices_B: ";
    PrintContentsOfVector<int>(indices_B);
    cout << "T: " << thread_id << " Frame " << frame_id << " indices_C: ";
    PrintContentsOfVector<int>(indices_C);
  }

  int index_B_ptr = 0;
  int index_C_ptr = 0;

  for (int index_B = 0; index_B < n_dimension_B; ++index_B) {

    // DEBUG_DEFENDER_FEATURE_SELECTION_LOOP
    // if (frame_id % 1000000 == 0 and index_B % 100 == 0) {
    //   cout << "T: " << thread_id << " index_B " << index_B << " Frame " << frame_id << " Debug: indices_B[" << index_B_ptr << "] = " << indices_B[index_B_ptr] << endl;
    // }

    if (index_B == 0)               regularizationU = Settings_->StageOne_Regularization_Weight_U_Level1;
    if (index_B == n_dimension_B)   regularizationU = Settings_->StageOne_Regularization_Weight_U_Level2;

    if (indices_B[index_B_ptr] < 0) {
      index_B_ptr += 1;
      continue;
    }

    if (index_B == indices_B[index_B_ptr]) {
      index_B_val = 1.0;
      index_B_ptr += 1;
    }
    else {
      index_B_val = 0.0;
    }

    index_C_ptr = 0;

    SKIP_OUT_OF_BOUNDS_INDICES_BH_S1

    // Loop over every defender grid cell
    for (int index_C = 0; index_C < n_dimension_C; ++index_C) {

      if (indices_C[index_C_ptr] < 0) {
        index_C_ptr += 1;
        index_C     -= 1;
        continue;
      }

      if (index_C == indices_C[index_C_ptr]) {
        index_C_val = 1.0;
        index_C_ptr += 1;  // Go to the next gridcell entry -- in Stage 1: goes to the position of the next defender
      }
      else {
        index_C_val = 0.0;
      }

      SKIP_OUT_OF_BOUNDS_INDICES_DEF_S1

      U_gradient = 0.;
      old_weight = Weight_U->at(index_A, index_B, index_C);

      // Gradient descent update
      U_gradient += strongweakweight * factor * index_B_val * index_C_val / n_MiniBatchSize;



      // Compute regularization
      if (Settings_->StageOne_Weight_U_RegularizationType == 1) {
        gradient_reg = GetLassoGradient(thread_id, old_weight);
      }
      else if (Settings_->StageOne_Weight_U_RegularizationType == 2) {
        gradient_reg = old_weight;
      }

      U_gradient += regularizationU * gradient_reg / n_MiniBatchSize;

      // Record the total gradient
      *(Weight_U_diff->att(index_A, index_B, index_C)) += U_gradient;

      if (Settings_->EnableDebugPrinter_Level1 and frame_id % 5000000 == 0 and index_B % 100 == 0 and U_gradient > 0) {
        cout << setw(15) << setprecision(10) << "Debug S1 gradients -- T: " << thread_id << " Frame " << frame_id << " index_B " << index_B << " index_C " << index_C << " indices_C[" << index_C_ptr << "] = " << indices_C[index_C_ptr];
        cout << setw(15) << setprecision(10) << " old_weight " << old_weight << " U_gradient "  << U_gradient << " factor " << factor << " index_B_val " << index_B_val << " index_C_val " << index_C_val << " regularizationU " << regularizationU << endl;
      }
    }
  }
}
void FullTensorModelTrainer::ProcessWeight_U_Updates(int thread_id, int index_A) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ProcessWeight_U_Updates");
  Settings  *Settings_               = this->Settings_;
  Tensor3Blob *Weight_U              = &(this->Weight_U);
  Tensor3Blob *Weight_U_diff         = &(this->Weight_U_Diff);
  Tensor3Blob *Weight_U_DiffSq_Cache = &(this->Weight_U_DiffSq_Cache);
  int n_dimension_B                  = Settings_->StageOne_Dimension_B;
  int n_dimension_C                  = Settings_->StageOne_Dimension_C;
  int n_dimension_A                  = Settings_->Dimension_A;
  int n_MiniBatchSize                = Settings_->StageOne_MiniBatchSize;

  float cum_sum_first_der            = 0.;
  float weight_update                = 0.;
  float weight_new                   = 0.;
  float adaptive_learning_rate       = 0.;

  float weight_old                   = 0.;

  assert(Settings_->StageOne_Weight_U_RegularizationType == 1 or Settings_->StageOne_Weight_U_RegularizationType == 2 or Settings_->StageOne_Weight_U_RegularizationType == 3);

  // cout << "Thread: " << thread_id << " ProcessWeight_U_Updates | index_A: " << index_A << endl;

  for (int index_B = 0; index_B < n_dimension_B; ++index_B) {

    for (int index_C = 0; index_C < n_dimension_C; ++index_C) {

      // cout << "Thread: " << thread_id << " index_B: " << index_B << " feat " << index_C << endl;

      // Adaptive learning rate: we divide by the sqrt sum of squared updates of the past.
      // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
      adaptive_learning_rate  = Settings_->StageOne_CurrentLearningRate;

      //ADAPTIVE_LEARNING_RATE
      if (Settings_->StageOne_UseAdaptiveLearningRate == 1) {
        cum_sum_first_der     = Weight_U_DiffSq_Cache->at(index_A, index_B, index_C);
        adaptive_learning_rate  = Settings_->StageOne_CurrentLearningRate / sqrt(cum_sum_first_der);

        if (adaptive_learning_rate > Settings_->StageOne_Clamp_AdaptiveLearningRate) {
          // PrintFancy(Settings_->session_start_time, "Learning rate update clamped! A: " + to_string(feat_index));
          adaptive_learning_rate = Settings_->StageOne_Clamp_AdaptiveLearningRate;
        }
      }

      weight_update = adaptive_learning_rate * Weight_U_diff->at(index_A, index_B, index_C);
      weight_old  = Weight_U->at(index_A, index_B, index_C);

      // Soft-thresholding: if the gradient takes us past 0, then just leave the weight at 0
      if (Settings_->StageOne_Weight_U_RegularizationType == 1) {

        if (Settings_->StageOne_Weight_U_UseSoftThreshold == 1) {
          *(Weight_U->att(index_A, index_B, index_C)) = SoftThresholdUpdate(thread_id, -1, weight_old, weight_update);
        } else {
          *(Weight_U->att(index_A, index_B, index_C)) -= weight_update;
        }

      } else if (Settings_->StageOne_Weight_U_RegularizationType == 2) {
        *(Weight_U->att(index_A, index_B, index_C)) -= weight_update;
      }

      if (Settings_->EnableDebugPrinter_Level1 == 1){
        if (thread_id == 0 and index_A == 0 and index_B % 10000 == 0 and index_C == 0) {

          if (debug_mode) {
            PrintDelimiter(0, 0, 80, '=');
            cout << setprecision(PRINT_FLOAT_PRECISION_SCORE) << std::setfill(' ') << "abc " << std::setw(9) << index_A << " " << index_B << " " << index_C;
            cout << " weight_old  " << std::setw(PRINT_FLOAT_PRECISION_SCORE+4) << weight_old;
            cout << " weight_update " << std::setw(PRINT_FLOAT_PRECISION_SCORE+4) << weight_update;
            cout << endl;
          }

        }
      }

      // if (Settings_->StageOne_Weight_U_UseSoftThreshold == 1) {
      //   if (Weight_U->at(feat_index, latent_index) < Settings_->StageOne_Weight_U_Threshold) {
      //     // if (feat_index == 0 and latent_index == 0) PrintFancy(Settings_->session_start_time, "Thresholding Weight U_A0! A: " + to_string(feat_index) + " weight: " + to_string(Weight_U->at(feat_index, latent_index)) );
      //     dirty_wghtU_threshold_counter++;
      //     Weight_U->at(feat_index, latent_index) = Settings_->StageOne_Weight_U_Threshold;
      //   }
      // }
    }
  }
}
void FullTensorModelTrainer::ComputeBias_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label) {
  // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeBias_A_Update");

  Settings  *Settings_       = this->Settings_;
  MatrixBlob  *ConditionalScore  = &this->ConditionalScore;
  VectorBlob  *Bias_A        = &this->Bias_A;
  VectorBlob  *Bias_A_Diff     = &this->Bias_A_Diff;
  VectorBlob  *Bias_A_DiffSq_Cache = &this->Bias_A_DiffSq_Cache;

  float ExponentialScore       = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float regularization_Bias_A    = Settings_->StageOne_Regularization_Bias_A;
  int   n_MiniBatchSize       = Settings_->StageOne_MiniBatchSize;

  // Multiplicity in data - there are multiple (#ground_truth_label) mentions of interaction for (A, J)
  float strongweakweight          = 0.;
  float factor                = 0.;

  if (ground_truth_label == 0) {
    strongweakweight = Settings_->LossWeightWeak;
    factor       = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakweight = Settings_->LossWeightStrong;
    factor       = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  float bias_A    = Bias_A->data[index_A];
  float bias_diff = (strongweakweight * factor + regularization_Bias_A * bias_A) / n_MiniBatchSize;

  *(Bias_A_Diff->att(index_A)) += bias_diff;
  // Bias_A_DiffSq_Cache->at(index_A)  += pow(bias_diff,2);
}
void FullTensorModelTrainer::ProcessBias_A_Updates(int thread_id, int index_A) {
  // // PrintFancy(Settings_->session_start_time, "T"+to_string(thread_id) + " - ComputeWeight_U_Update");
  Settings      *Settings_       = this->Settings_;
  VectorBlob      *Bias_A        = &this->Bias_A;
  VectorBlob      *Bias_A_Diff     = &this->Bias_A_Diff;
  VectorBlob      *Bias_A_DiffSq_Cache = &this->Bias_A_DiffSq_Cache;
  int         n_MiniBatchSize     = Settings_->StageOne_MiniBatchSize;

  float cum_sum_first_der          = Bias_A_DiffSq_Cache->at(index_A);

  // Adaptive learning rate        : we divide by the sqrt sum of squared updates of the past.
  // Note that we clamp this number, 1 / sqrt(small number) --> dangerous
  float adaptive_learning_rate       = Settings_->StageOne_CurrentLearningRate;

  // ADAPTIVE_LEARNING_RATE
  if (Settings_->StageOne_UseAdaptiveLearningRate == 1) {
    adaptive_learning_rate  = Settings_->StageOne_CurrentLearningRate / sqrt(cum_sum_first_der);
    if (index_A == 0) cout << "Adaptive LearningRate: bias A:" << index_A << " :: " << Settings_->StageOne_CurrentLearningRate << " -> " << adaptive_learning_rate << " using cum_sum_first_der = " << cum_sum_first_der << endl;
    if (adaptive_learning_rate > Settings_->StageOne_Clamp_AdaptiveLearningRate) {
      adaptive_learning_rate = Settings_->StageOne_Clamp_AdaptiveLearningRate;
    }
  }
  *(Bias_A->att(index_A)) -= adaptive_learning_rate * Bias_A_Diff->at(index_A);
}


void FullTensorModelTrainer::Store_Snapshot(int cross_val_run, int epoch , string fp_snapshot) {

  Settings *Settings_         = this->Settings_;
  CurrentStateBlob *CurrentStateBlob_ = this->CurrentStateBlob_;

  // Find index of last recorded loss
  int current_iteration = cross_val_run * Settings_->StageOne_NumberOfCrossValidationRuns + epoch;
  int n_losses_were_recorded = static_cast<int>(floor(static_cast<float>(current_iteration) / Settings_->StageOne_ComputeLossAfterEveryNthEpoch));

  int last_recorded_loss_index = this->Results_LossTrain.size() - 1;

  Tensor3Blob     *tt;
  MatrixBlob      *mm;
  std::vector<float>  *ff;
  VectorBlob      *vvv;

  cout << setw(10) << "DEBUG:" << this->Results_LossTrain.size() << endl;

  time_t now = time(0);
  char timestamp[80];
  strftime(timestamp, sizeof(timestamp), "%d-%b-%Y_%H-%M-%S", localtime(&now));
  string  fn;
  FILE  *fp;

  // We write the current settings ONCE per program run
  if (CurrentStateBlob_->curr_snapshot_id == 0) {
    fn = fp_snapshot + \
          "/rank3/stage1/" + Settings_->DATASET_NAME + "_r3s1_"+ \
          "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" +  \
          "|_lr_" + to_string(Settings_->StageOne_StartingLearningRate) + "_" + \
          "|_rUtype_" + to_string(Settings_->StageOne_Weight_U_RegularizationType) + "_" + \
          "|_rU_" + to_string(Settings_->StageOne_Regularization_Weight_U_Level1) + "_" + \
          ".override_settings";
    fp = fopen( fn.c_str() , "wb");
    const char c = '!';
    fwrite(&c, sizeof(char), 1, fp);
    fclose(fp);

    // // Copy of settings.json -- mainly use to see what snapshot we started from
    // string fp_settings      = fp_local_cpp + "/settings.json";
    // string fp_settings_snapshot = fp_snapshot + "/rank3/stage1/ss_s1_rank3_" + \
    //                 "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
    //                 to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
    //                 "settings" + "_" + \
    //                 to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
    //                 to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
    //                 to_string(this->Results_LossTest[last_recorded_loss_index]) + ".json";

    // std::ifstream  src(fp_settings.c_str(),      std::ios::binary);
    // std::ofstream  dst(fp_settings_snapshot.c_str(), std::ios::binary);

    // dst << src.rdbuf();

    // Shuffled indices -- how do we loop throuth the groundtruthlabels?
    fn = fp_snapshot + \
          "/rank3/stage1/" + Settings_->DATASET_NAME + "_r3s1_"+ \
          "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
          to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
          "shuffled_train_val_indices" + "_" + \
          to_string(cross_val_run) + "_" + \
          to_string(epoch) + "_" + \
          ".bin";

    fp = fopen( fn.c_str() , "wb");

    std::vector<int> *vv = &this->CurrentTrainIndices_Strong;
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
        "/rank3/stage1/" + Settings_->DATASET_NAME + "_r3s1_"+ \
        "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
        to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
        "tvt+loss" + "_" + \
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
      "/rank3/stage1/" + Settings_->DATASET_NAME + "_r3s1_"+ \
      "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
      to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
      "wgt+bias" + "_" + \
      to_string(cross_val_run) + "_" + \
      to_string(epoch) + "_" + \
      to_string(this->Results_LossTrain[last_recorded_loss_index]) + "_" +\
      to_string(this->Results_LossValid[last_recorded_loss_index]) +\
      ".bin";

  fp = fopen( fn.c_str(), "wb");

  tt = &this->Weight_U;
  for (int i = 0; i < tt->data.size(); ++i) fwrite(&tt->data[i], sizeof(float), 1, fp);
  // tt = &this->Weight_U_DiffSq_Cache;
  // for (int i = 0; i < tt->data.size(); ++i) fwrite(&tt->data[i], sizeof(float), 1, fp);

  vvv = &this->Bias_A;
  for (int i = 0; i < vvv->data.size(); ++i) fwrite(&vvv->data[i], sizeof(float), 1, fp);
  // vvv = &this->Bias_A_DiffSq_Cache;
  // for (int i = 0; i < vvv->data.size(); ++i) fwrite(&vvv->data[i], sizeof(float), 1, fp);

  fclose(fp);

  // Write ExponentialScores to file
  // fn =  fp_snapshot + \
  //     "/rank3/stage1/" + Settings_->DATASET_NAME + "_r3s1_"+ \
  //     "__PYID___" + Settings_->PYTHON_LOGGING_ID + "_" + \
  //     to_string(CurrentStateBlob_->curr_snapshot_id) + "_" + \
  //     "ExponentialScorelog" + "_" + \
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


