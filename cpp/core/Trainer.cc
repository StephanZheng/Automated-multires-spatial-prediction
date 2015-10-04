// #include <boost/chrono.hpp>
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

#include "core/Trainer.h"

using namespace std;
using namespace boost;
using namespace rapidjson;
using namespace std::chrono;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

int Trainer::CheckIfFloatIsNan(float x, string y) {
  if (x != x || boost::math::isinf(abs(x)) || boost::math::isnan(abs(x))) {
    PrintFancy(Settings_->session_start_time, "NaN! Info: " + y + " | " + to_string(x));
    return 1;
  } else {
    return 0;
  }
}

int Trainer::generateTrainingBatches(int task_type, int batch, int n_batches, int n_datapoints_processed_so_far_this_epoch, int MiniBatchSize) {

  // PrintFancy(Settings_->session_start_time, "generateTrainingBatches - "+to_string(task_type) + " batch: " + to_string(batch));

  Settings     *Settings_                            = this->Settings_;
  GroundTruthLabel *GroundTruthLabelsTrainValWeak_   = this->GroundTruthLabelsTrainValWeak_;
  GroundTruthLabel *GroundTruthLabelsTrainValStrong_ = this->GroundTruthLabelsTrainValStrong_;
  std::vector<int> *CurrentTrainIndices_Strong       = &this->CurrentTrainIndices_Strong;
  std::vector<int> *CurrentTrainIndices_Weak         = &this->CurrentTrainIndices_Weak;
  std::vector<int> *RandomizedIterator               = &this->RandomizedIterator;

  int current_iterator                               = 0;

  int n_total_labels_train                           = Settings_->NumberOfWeakLabels_Train + Settings_->NumberOfStrongLabels_Train;
  assert(n_datapoints_processed_so_far_this_epoch < n_total_labels_train);

  int n_mini_batches_dispatched_this_batch           = 0;  // Hoy many do we see actually (last batch)
  int current_label;

  int index_datapoint_to_process                     = n_datapoints_processed_so_far_this_epoch;
  int n_datapoints_this_iteration                    = 0;

  QueueMessage *QueueMessageSend_MiniBatch       = new QueueMessage[MiniBatchSize];

  // For every thread, we create a set of tasks
  for (int thread = 0; thread < Settings_->CurrentNumberOfThreads; ++thread) {
    // For every entry in the minibatch, create a QueueMessage
    for (int mini_batch_entry = 0; mini_batch_entry < MiniBatchSize; ++mini_batch_entry) {

      // We're past the last datapoint: fill remaining tasks with dummies and exit
      if (batch == n_batches-1 and index_datapoint_to_process >= n_total_labels_train) {

        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = -1;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = -1;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = -1;

        // cout << index_datapoint_to_process << mini_batch_entry << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].index_A  << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].frame_id      << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label  << endl;

        continue;
      }

      // Construct task message
      QueueMessageSend_MiniBatch[mini_batch_entry].task_type       = task_type;

      assert(index_datapoint_to_process < RandomizedIterator->size());
      current_iterator = RandomizedIterator->at(index_datapoint_to_process);
      assert(current_iterator < Settings_->NumberOfWeakLabels_Train + Settings_->NumberOfStrongLabels_Train);

      if (current_iterator < Settings_->NumberOfWeakLabels_Train) {
        current_label = CurrentTrainIndices_Weak->at(current_iterator);
        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = GroundTruthLabelsTrainValWeak_[current_label].index_A;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = GroundTruthLabelsTrainValWeak_[current_label].frame_id;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = GroundTruthLabelsTrainValWeak_[current_label].ground_truth_label;
      }
      if (current_iterator >= Settings_->NumberOfWeakLabels_Train) {
        // Be careful: index_datapoint_to_process is an index in weak+strong labels, so we need to fix the right starting point for the strong labels
        current_label = CurrentTrainIndices_Strong->at(current_iterator - Settings_->NumberOfWeakLabels_Train);
        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = GroundTruthLabelsTrainValStrong_[current_label].index_A;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = GroundTruthLabelsTrainValStrong_[current_label].frame_id;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = GroundTruthLabelsTrainValStrong_[current_label].ground_truth_label;
      }

      n_datapoints_this_iteration++;
      index_datapoint_to_process++;
      // The last batch might be smaller that the number of threads, so break early.
    }

    // Put in queue
    task_queue_.mutex_.lock();
    for (int i = 0; i < MiniBatchSize; ++i) {
      task_queue_.taskQueue.push(QueueMessageSend_MiniBatch[i]);
    }
    // PrintFancy(Settings_->session_start_time, "Queue has size " + to_string(taskQueue.size()));
    // PrintFancy(Settings_->session_start_time, "Served     " + to_string(n_datapoints_processed_so_far_this_epoch));
    task_queue_.mutex_.unlock();

    task_queue_.qGoFetch.notify_one();
    n_mini_batches_dispatched_this_batch++;

    // PrintFancy(Settings_->session_start_time, "End of minibatch run: we've seen #datapoints        : " + to_string(n_datapoints_this_iteration));
    // PrintFancy(Settings_->session_start_time, "End of minibatch run: index_datapoint_to_process is now at: " + to_string(index_datapoint_to_process));

    if (batch == n_batches-1 and index_datapoint_to_process >= n_total_labels_train) {
      if (task_type == 1) PrintFancy(Settings_->session_start_time, "Total #train data: " + to_string(index_datapoint_to_process - 1));
      cout << "Debug: n_total_labels_train: " << n_total_labels_train << " index_datapoint_to_process " << index_datapoint_to_process << " n_mini_batches_dispatched_this_batch " << n_mini_batches_dispatched_this_batch << endl;
      break;
    }
  }


  delete[] QueueMessageSend_MiniBatch;

  // Note that we don't need to tell the main routine how many items were in the last batch,
  // because we will never generate more batches anyway this epoch
  return n_mini_batches_dispatched_this_batch;
}
int Trainer::generateValidationBatches(int task_type, int batch, int n_batches, int n_datapoints_processed_so_far_this_epoch, int MiniBatchSize) {
  Settings     *Settings_                            = this->Settings_;
  GroundTruthLabel *GroundTruthLabelsTrainValWeak_   = this->GroundTruthLabelsTrainValWeak_;
  GroundTruthLabel *GroundTruthLabelsTrainValStrong_ = this->GroundTruthLabelsTrainValStrong_;
  std::vector<int> *CurrentValidIndices_Strong       = &this->CurrentValidIndices_Strong;
  std::vector<int> *CurrentValidIndices_Weak         = &this->CurrentValidIndices_Weak;

  int n_total_labels_valid                           = Settings_->NumberOfWeakLabels_Val + Settings_->NumberOfStrongLabels_Val;
  assert(n_datapoints_processed_so_far_this_epoch < n_total_labels_valid);

  int index_datapoint_to_process                     = n_datapoints_processed_so_far_this_epoch;
  int n_datapoints_this_iteration                    = 0;

  int n_mini_batches_dispatched_this_batch           = 0;                   // Hoy many do we see actually (last batch)
  int current_label                                  = 0;

  QueueMessage *QueueMessageSend_MiniBatch           = new QueueMessage[MiniBatchSize];

  // PrintFancy(Settings_->session_start_time, n_datapoints_processed_so_far_this_epoch);

  // For every thread, we create a set of tasks
  for (int thread = 0; thread < Settings_->CurrentNumberOfThreads; ++thread) {
    // For every entry in the minibatch, create a QueueMessage
    for (int mini_batch_entry = 0; mini_batch_entry < MiniBatchSize; ++mini_batch_entry) {

      // We're past the last datapoint: fill remaining tasks with dummies and exit
      if (batch == n_batches-1 and index_datapoint_to_process >= n_total_labels_valid) {

        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = -1;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = -1;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = -1;

        // cout << index_datapoint_to_process << mini_batch_entry << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].index_A  << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].frame_id      << endl;
        // cout << QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label  << endl;

        continue;
      }

      // Construct task message
      QueueMessageSend_MiniBatch[mini_batch_entry].task_type       = task_type;

      if (index_datapoint_to_process < Settings_->NumberOfWeakLabels_Val) {
        current_label = CurrentValidIndices_Weak->at(index_datapoint_to_process); // n_datapoints_processed_so_far_this_epoch + thread * MiniBatchSize + mini_batch_entry);
        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = GroundTruthLabelsTrainValWeak_[current_label].index_A;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = GroundTruthLabelsTrainValWeak_[current_label].frame_id;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = GroundTruthLabelsTrainValWeak_[current_label].ground_truth_label;
        }

      if (index_datapoint_to_process >= Settings_->NumberOfWeakLabels_Val) {
        // Be careful: n_datapoints_processed_so_far_this_epoch is an index in weak+strong labels, so we need to fix the right starting point for the strong labels
        current_label = CurrentValidIndices_Strong->at(index_datapoint_to_process - Settings_->NumberOfWeakLabels_Val);
        QueueMessageSend_MiniBatch[mini_batch_entry].index_A      = GroundTruthLabelsTrainValStrong_[current_label].index_A;
        QueueMessageSend_MiniBatch[mini_batch_entry].frame_id       = GroundTruthLabelsTrainValStrong_[current_label].frame_id;
        QueueMessageSend_MiniBatch[mini_batch_entry].ground_truth_label = GroundTruthLabelsTrainValStrong_[current_label].ground_truth_label;
      }

      n_datapoints_this_iteration++;
      index_datapoint_to_process++;

      // The last batch might be smaller that the number of threads, so break early.
      if (batch == n_batches-1 and index_datapoint_to_process >= n_total_labels_valid) break;
    }

    // Put in queue
    task_queue_.mutex_.lock();
    for (int i = 0; i < MiniBatchSize; ++i) {
      task_queue_.taskQueue.push(QueueMessageSend_MiniBatch[i]);
    }
    task_queue_.mutex_.unlock();
    task_queue_.qGoFetch.notify_one();
    n_mini_batches_dispatched_this_batch++;

    // PrintFancy(Settings_->session_start_time, "End of minibatch run: we've seen #datapoints        : " + to_string(n_datapoints_this_iteration));
    // PrintFancy(Settings_->session_start_time, "End of minibatch run: index_datapoint_to_process is now at: " + to_string(index_datapoint_to_process));

    if (batch == n_batches-1 and index_datapoint_to_process >= n_total_labels_valid) {
      if (task_type == 1) PrintFancy(Settings_->session_start_time, "Total #validation data: " + to_string(index_datapoint_to_process - 1));
      break;
    }
  }

  delete[] QueueMessageSend_MiniBatch;

  // Note that we don't need to tell the main routine how many items were in the last batch,
  // because we will never generate more batches anyway this epoch
  return n_mini_batches_dispatched_this_batch;
}

void Trainer::PermuteTrainValIndices() {
  PrintFancy(Settings_->session_start_time, "Permuting train / valid split");

  Settings *Settings_                                 = this->Settings_;
  std::vector<int> *CurrentTrainIndices_Strong        = &this->CurrentTrainIndices_Strong;
  std::vector<int> *CurrentValidIndices_Strong        = &this->CurrentValidIndices_Strong;
  std::vector<int> *CurrentTrainIndices_Weak          = &this->CurrentTrainIndices_Weak;
  std::vector<int> *CurrentValidIndices_Weak          = &this->CurrentValidIndices_Weak;

  std::vector<int> *CurrentTrainIndices_Strong_Sorted = &this->CurrentTrainIndices_Strong_Sorted;
  std::vector<int> *CurrentTrainIndices_Weak_Sorted   = &this->CurrentTrainIndices_Weak_Sorted;

  // Copy the old validation vector into a temp one

  std::vector<int> temp_val_strong(Settings_->NumberOfStrongLabels_Val);
  std::vector<int> temp_val_weak(Settings_->NumberOfWeakLabels_Val);
  temp_val_strong                                     = *CurrentValidIndices_Strong;
  temp_val_weak                                       = *CurrentValidIndices_Weak;

  // The new validation set is the first N entries in the old train-vector
  for (int i                                          = 0; i < CurrentValidIndices_Strong->size(); ++i)  CurrentValidIndices_Strong->at(i)   = CurrentTrainIndices_Strong->at(i);
  for (int i                                          = 0; i < CurrentValidIndices_Weak->size(); ++i)    CurrentValidIndices_Weak->at(i)   = CurrentTrainIndices_Weak->at(i);

  cout << "Debug Permute" << endl;
  cout << "CurrentTrainIndices_Strong.size(): " << CurrentTrainIndices_Strong->size() << endl;
  cout << "Settings_->NumberOfStrongLabels_Val: " << Settings_->NumberOfStrongLabels_Val << endl;

  // Remove the first N entries from the old vector and push the old validation on
  CurrentTrainIndices_Strong->erase(CurrentTrainIndices_Strong->begin(),   CurrentTrainIndices_Strong->begin() + Settings_->NumberOfStrongLabels_Val);
  CurrentTrainIndices_Strong->insert(CurrentTrainIndices_Strong->end(), temp_val_strong.begin(), temp_val_strong.end());

  CurrentTrainIndices_Weak->erase(CurrentTrainIndices_Weak->begin(),   CurrentTrainIndices_Weak->begin() + Settings_->NumberOfWeakLabels_Val);
  CurrentTrainIndices_Weak->insert(CurrentTrainIndices_Weak->end(), temp_val_weak.begin(), temp_val_weak.end());

  CurrentTrainIndices_Strong_Sorted->resize(0);
  CurrentTrainIndices_Strong_Sorted->insert(CurrentTrainIndices_Strong_Sorted->begin(), CurrentTrainIndices_Strong->begin(), CurrentTrainIndices_Strong->end());
  sort(CurrentTrainIndices_Strong_Sorted->begin(), CurrentTrainIndices_Strong_Sorted->end());

  CurrentTrainIndices_Weak_Sorted->resize(0);
  CurrentTrainIndices_Weak_Sorted->insert(CurrentTrainIndices_Weak_Sorted->begin(), CurrentTrainIndices_Weak->begin(), CurrentTrainIndices_Weak->end());
  sort(CurrentTrainIndices_Weak_Sorted->begin(), CurrentTrainIndices_Weak_Sorted->end());

  // Sanity check
  assert(CurrentTrainIndices_Strong->size()           == Settings_->NumberOfStrongLabels_Train);
  assert(CurrentTrainIndices_Strong_Sorted->size()    == Settings_->NumberOfStrongLabels_Train);
  assert(CurrentTrainIndices_Weak->size()             == Settings_->NumberOfWeakLabels_Train);
  assert(CurrentTrainIndices_Weak_Sorted->size()      == Settings_->NumberOfWeakLabels_Train);
  assert(CurrentValidIndices_Strong->size()           == Settings_->NumberOfStrongLabels_Val);
  assert(CurrentValidIndices_Weak->size()             == Settings_->NumberOfWeakLabels_Val);
}
void Trainer::PrintStatus() {

  Settings *Settings_ = this->Settings_;
  string ub;
  string alr;
  string lf;
  string sms;
  if (Settings_->StageThree_UseBias == 1) {
    ub = "ON";
  } else {
    ub = "OFF";
  }
  if (Settings_->StageThree_UseAdaptiveLearningRate == 1) {
    alr = "ON";
  } else {
    alr = "OFF";
  }
  if (Settings_->StageThree_UseLatentTerm == 1) {
    lf = "ON";
  } else {
    lf = "OFF";
  }
  if (Settings_->StageThree_UseSparseMatS == 1) {
    sms = "ON";
  } else {
    sms = "OFF";
  }

  cout << endl;
  PrintFancy(Settings_->session_start_time, "Hi there! Settings of this run\n");
  PrintFancy(Settings_->session_start_time, "Momentum update        : ON");
  PrintFancy(Settings_->session_start_time, "Adaptive learning rate : "+alr);
  PrintFancy(Settings_->session_start_time, "Stochastic             : ON");
  PrintFancy(Settings_->session_start_time, "Bias                   : "+ub);
  PrintFancy(Settings_->session_start_time, "LF                     : "+lf);
  PrintFancy(Settings_->session_start_time, "SparseMatS             : "+sms);

  cout << endl;
}
// Dot product in two column spaces
float Trainer::MatrixBlobDotProduct(int thread_id,
  int index_1,
  int index_2,
  MatrixBlob *Matrix_A,
  MatrixBlob *Matrix_B
  ) {
  assert(Matrix_A->columns == Matrix_B->columns);
  float sum = 0.;
  for (int col = 0; col < Matrix_A->columns; col++) {
    sum += Matrix_A->at(index_1, col) * Matrix_B->at(index_2, col);
  }
  return sum;
}

// Three-way dot product in three column spaces!
float Trainer::MatrixBlobDotProductThreeWay(int thread_id,
  int index_1,
  int index_2,
  int index_3,
  MatrixBlob *Matrix_A,
  MatrixBlob *Matrix_B,
  MatrixBlob *Matrix_C
  ) {
  assert(Matrix_A->columns == Matrix_B->columns);
  assert(Matrix_A->columns == Matrix_C->columns);
  float sum = 0.;
  for (int col = 0; col < Matrix_A->columns; col++) {
    sum += Matrix_A->at(index_1, col) * Matrix_B->at(index_2, col) * Matrix_C->at(index_3, col);
  }
  return sum;
}
std::vector<int> Trainer::getIndices_B(int frame_id, int option) {

  MatrixBlob  *Blob_B  = this->Blob_B_;
  MatrixBlob  *Blob_BB   = this->Blob_BB_;
  Settings  *Settings_ = this->Settings_;

  assert (option == 1 or option == 3 or option == 31 or option == 32);

  // Note that we add a sentinel value at the end of indices_B. If we have seen all defenders,
  // the loop over all non-zero features should still update all remaining fields.

  int size = 0;
  if (option == 1) {
    size = Settings_->StageOne_NumberOfNonZeroEntries_B + 1;
  } else if (option == 3) {
    size = Settings_->StageThree_NumberOfNonZeroEntries_B + 1;
  } else if (option == 31) {
    size = Settings_->StageThree_NumberOfNonZeroEntries_B * NUMBER_OF_INTS_WINDOW_3BY3 + 1;
  } else size = 0;

  std::vector<int> indices_B(size, 0);

  for (int i = 0; i < size - 1; ++i)
  {
    if (option == 1 or option == 3){
      indices_B[i] = static_cast<int>(Blob_B->at(frame_id, i));
    } else if (option == 31) {
      indices_B[i] = static_cast<int>(Blob_BB->at(frame_id, i));
    }
  }

  // The sentinel value is some offset above the maximal value that the feature can attain
  if (option == 1) {
    indices_B[indices_B.size() - 1] = Settings_->StageOne_Dimension_B + SENTINEL_OFFSET;
  } else if (option == 3 or option == 31) {
    indices_B[indices_B.size() - 1] = Settings_->StageThree_Dimension_B + SENTINEL_OFFSET;
  } else size = 0;

  return indices_B;
}
std::vector<int> Trainer::getIndices_C(int frame_id, int option) {

  MatrixBlob  *Blob_C  = this->Blob_C_;
  MatrixBlob  *Blob_CC   = this->Blob_CC_;
  Settings  *Settings_ = this->Settings_;
  assert (option == 1 or option == 3 or option == 31 or option == 32);

  // Note that we add a sentinel value at the end of indices_C. If we have seen all defenders,
  // the loop over all non-zero features should still update all remaining fields.

  int size = 0;
  if (option == 1) {
    size = Settings_->DummyMultiplier * (Settings_->StageOne_NumberOfNonZeroEntries_C) + 1;
  } else if (option == 3) {
    size = Settings_->DummyMultiplier * (Settings_->StageThree_NumberOfNonZeroEntries_C) + 1;
  } else if (option == 31) {
    size = Settings_->DummyMultiplier * ((Settings_->StageThree_NumberOfNonZeroEntries_C - Settings_->Dimension_C_BiasSlice) * NUMBER_OF_INTS_WINDOW_3BY3) + 1;
  } else size = 0;

  std::vector<int> indices_C(size, 0);

  // Fill all entries but one of C with the feature values
  if (option == 1 or option == 3){
    for (int i = 0; i < size - 1; ++i)
    {
      indices_C[i] = static_cast<int>(Blob_C->at(frame_id, i) );
    }
  } else if (option == 31) {
    for (int i = 0; i < size - 1; ++i)
    {
      indices_C[i] = static_cast<int>(Blob_CC->at(frame_id, i) );
    }
  }

  // The sentinel value is some offset above the maximal value that the feature can attain
  if (option == 1) {
    indices_C[indices_C.size() - 1] = Settings_->StageOne_Dimension_C + SENTINEL_OFFSET;
  } else if (option == 3 or option == 31) {
    indices_C[indices_C.size() - 1] = Settings_->StageThree_Dimension_C + SENTINEL_OFFSET;
  } else size = 0;

  return indices_C;
}
void Trainer::BoostParameter(int batch, Tensor3Blob *parameter, Tensor3Blob *parameter_diff, Tensor3Blob *parameter_DiffSq_Cache, std::vector<float>  *NestorovMomentumPreviousParameter_, int n_elements) {

  Settings      *Settings_         = this->Settings_;
  std::vector<float>  *NestorovMomentumLambda_ = &this->NestorovMomentumLambda;

  // PrintFancy(Settings_->session_start_time, "Applying momentum");
  // int n_dimension_A              = Settings_->Dimension_A;
  // int nOccupancyFeatures             = Settings_->Dimension_B;
  // int n_dimension_latent          = Settings_->NumberOfLatentDimensions;

  // Boost parameters
  float change_this_round            = 0.;
  float cum_sum_first_der            = 0.;
  float new_first_der              = 0.;

  float parameter_prev_round           = 0.;
  float parameter_this_round           = 0.;
  float parameter_updated            = 0.;

  float lambda                 = 0.;
  float lambda_new               = 0.;
  float gamma                  = 0.;

  // When this method is called, we've updated lambda already in the method ApplyNestorovMomentum.
  int n_momentum_updates_seen = NestorovMomentumLambda_->size();
  lambda    = NestorovMomentumLambda_->at(n_momentum_updates_seen - 2);
  lambda_new  = NestorovMomentumLambda_->at(n_momentum_updates_seen - 1);
  gamma     = (1 - lambda) / lambda_new;

  if (Settings_->EnableDebugPrinter_Level1 == 1) PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) + " -- TensorBlob -- applying momentum updates with gamma: " + to_string(gamma));

  for (int i = 0; i < n_elements; ++i) {
    if (Settings_->StageOne_UseAdaptiveLearningRate == 1) {
      cum_sum_first_der  = parameter_DiffSq_Cache->data[i];
    } else {
      cum_sum_first_der  = 1.;
    }

    // Note that the update rule was parameter_this_round = parameter_prev_round - lrate * parameter_diff / adaptscale
    parameter_this_round = parameter->data[i];  // parameter contains the updated at time s+1. Note the clash of names: we'll rename it in the future
    parameter_prev_round = NestorovMomentumPreviousParameter_->at(i);  // gradient desecnt subtracts the first der

    // Now we have the needed entries, update the parameter AND the stored diff sq cache
    // Note that parameter_diff is deleted and recalculated in the next minibatch anyway, so we don't care
    parameter_updated  = (1-gamma) * parameter_this_round + gamma * parameter_prev_round;
    parameter->data[i]   = parameter_updated;

    // Store the updated parameter, for the next momentum update
    NestorovMomentumPreviousParameter_->at(i) = parameter_updated;

    // Update the cache: subtract the old diff^2 and add the new diff^2
    // Note: back out what the new first derivative would be, add that to the cache
    new_first_der           = sqrt(cum_sum_first_der) * (parameter_updated - parameter_this_round + parameter_diff->data[i]) / Settings_->StageOne_CurrentLearningRate;
    parameter_DiffSq_Cache->data[i] += pow(new_first_der, 2) - pow(parameter_diff->data[i], 2);
  }
}
void Trainer::BoostParameter(int batch, MatrixBlob *parameter, MatrixBlob *parameter_diff, MatrixBlob *parameter_DiffSq_Cache, std::vector<float>  *NestorovMomentumPreviousParameter_, int n_elements) {
  Settings      *Settings_         = this->Settings_;
  std::vector<float>  *NestorovMomentumLambda_ = &this->NestorovMomentumLambda;

  // Boost parameters
  float change_this_round            = 0.;
  float cum_sum_first_der            = 0.;
  float new_first_der              = 0.;

  float parameter_prev_round           = 0.;
  float parameter_this_round           = 0.;
  float parameter_updated            = 0.;

  float lambda                 = 0.;
  float lambda_new               = 0.;
  float gamma                  = 0.;

  // When this method is called, we've updated lambda already in the method ApplyNestorovMomentum.
  int n_momentum_updates_seen = NestorovMomentumLambda_->size();

  assert (NestorovMomentumLambda_->size() >= 2);

  lambda    = NestorovMomentumLambda_->at(n_momentum_updates_seen - 2);
  lambda_new  = NestorovMomentumLambda_->at(n_momentum_updates_seen - 1);
  gamma   = (1 - lambda) / lambda_new;

  if (Settings_->EnableDebugPrinter_Level1 == 1) PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) + " -- MatrixBlob -- applying momentum updates with gamma: " + to_string(gamma));

  for (int i = 0; i < n_elements; ++i) {
    if (Settings_->StageOne_UseAdaptiveLearningRate == 1) {
      cum_sum_first_der  = parameter_DiffSq_Cache->data[i];
    } else {
      cum_sum_first_der  = 1.;
    }

    // Note that the update rule was parameter_this_round = parameter_prev_round - lrate * parameter_diff / adaptscale
    parameter_this_round = parameter->data[i];  // parameter contains the updated at time s+1. Note the clash of names: we'll rename it in the future
    parameter_prev_round = NestorovMomentumPreviousParameter_->at(i);  // gradient desecnt subtracts the first der

    // Now we have the needed entries, update the parameter AND the stored diff sq cache
    // Note that parameter_diff is deleted and recalculated in the next minibatch anyway, so we don't care
    parameter_updated  = (1-gamma) * parameter_this_round + gamma * parameter_prev_round;
    parameter->data[i] = parameter_updated;

    // Store the updated parameter, for the next momentum update
    NestorovMomentumPreviousParameter_->at(i) = parameter_updated;

    // if ((i == 100 || i == 200 || i == 300 || i == 400 ) and batch % 100 == 0) cout << "Momentum gamma_s: " << gamma << " | parameter"+to_string(i)+",0: new-old: " << parameter_this_round - parameter_prev_round << " :: update " << parameter_this_round << " -> " << parameter->data[i] << endl;

    // Update the cache: subtract the old diff^2 and add the new diff^2
    // Note: back out what the new first derivative would be, add that to the cache
    new_first_der           = sqrt(cum_sum_first_der) * (parameter_updated - parameter_this_round + parameter_diff->data[i]) / Settings_->StageOne_CurrentLearningRate;
    parameter_DiffSq_Cache->data[i] += pow(new_first_der, 2) - pow(parameter_diff->data[i], 2);
  }
}
void Trainer::BoostParameter(int batch, VectorBlob *parameter, VectorBlob *parameter_diff, VectorBlob *parameter_DiffSq_Cache, std::vector<float> *NestorovMomentumPreviousParameter_, int n_elements) {

  Settings      *Settings_         = this->Settings_;
  std::vector<float>  *NestorovMomentumLambda_ = &this->NestorovMomentumLambda;

  // Boost parameters
  float change_this_round            = 0.;
  float cum_sum_first_der            = 0.;
  float new_first_der              = 0.;

  float parameter_prev_round           = 0.;
  float parameter_this_round           = 0.;
  float parameter_updated            = 0.;

  float lambda                 = 0.;
  float lambda_new               = 0.;
  float gamma                  = 0.;

  // When this method is called, we've updated lambda already in the method ApplyNestorovMomentum.
  int n_momentum_updates_seen = NestorovMomentumLambda_->size();
  lambda    = NestorovMomentumLambda_->at(n_momentum_updates_seen - 2);
  lambda_new  = NestorovMomentumLambda_->at(n_momentum_updates_seen - 1);
  gamma     = (1 - lambda) / lambda_new;

  if (Settings_->EnableDebugPrinter_Level1 == 1) PrintFancy(Settings_->session_start_time, "Batch " + to_string(batch) + " -- VectorBlob -- applying momentum updates with gamma: " + to_string(gamma));

  for (int i = 0; i < n_elements; ++i) {
    if (Settings_->StageOne_UseAdaptiveLearningRate == 1) {
      cum_sum_first_der  = parameter_DiffSq_Cache->data[i];
    } else {
      cum_sum_first_der  = 1.;
    }

    // Note that the update rule was parameter_this_round = parameter_prev_round - lrate * parameter_diff / adaptscale
    parameter_this_round = parameter->data[i];  // parameter contains the updated at time s+1. Note the clash of names: we'll rename it in the future
    parameter_prev_round = NestorovMomentumPreviousParameter_->at(i);  // gradient desecnt subtracts the first der

    // Now we have the needed entries, update the parameter AND the stored diff sq cache
    // Note that parameter_diff is deleted and recalculated in the next minibatch anyway, so we don't care
    parameter_updated  = (1-gamma) * parameter_this_round + gamma * parameter_prev_round;
    parameter->data[i]   = parameter_updated;

    // Store the updated parameter, for the next momentum update
    NestorovMomentumPreviousParameter_->at(i) = parameter_updated;

    // Update the cache: subtract the old diff^2 and add the new diff^2
    // Note: back out what the new first derivative would be, add that to the cache
    new_first_der               = sqrt(cum_sum_first_der) * (parameter_updated - parameter_this_round + parameter_diff->data[i]) / Settings_->StageOne_CurrentLearningRate;
    parameter_DiffSq_Cache->data[i]   += pow(new_first_der, 2) - pow(parameter_diff->data[i], 2);
  }
}
void Trainer::ComputeConditionalScoreSingleThreaded(int type) {
  // Use to compute ExponentialScore for loss_valid, loss_test

  Settings *Settings_      = this->Settings_;
  MatrixBlob *ConditionalScore = &(this->ConditionalScore);

  float   score        = 0.;
  float   gtruth_label     = 0.;

  int   index_A        = 0;
  int   frame_id       = 0;
  int   ground_truth_label   = 0;

  GroundTruthLabel *GroundTruthLabelsTrainValWeak_  = this->GroundTruthLabelsTrainValWeak_;
  GroundTruthLabel *GroundTruthLabelsTrainValStrong_  = this->GroundTruthLabelsTrainValStrong_;

  std::vector<int> *CurrentValidIndices_Strong    = &this->CurrentValidIndices_Strong;
  std::vector<int> *CurrentValidIndices_Weak      = &this->CurrentValidIndices_Weak;

  GroundTruthLabel *GroundTruthLabelsTestWeak_    = this->GroundTruthLabelsTestWeak_;
  GroundTruthLabel *GroundTruthLabelsTestStrong_    = this->GroundTruthLabelsTestStrong_;

  if (type == 7) {
    for (int &weak_label_valid : *CurrentValidIndices_Weak) {
      index_A      = GroundTruthLabelsTrainValWeak_[weak_label_valid].index_A;
      frame_id       = GroundTruthLabelsTrainValWeak_[weak_label_valid].frame_id;
      ground_truth_label = GroundTruthLabelsTrainValWeak_[weak_label_valid].ground_truth_label;

      score = this->ComputeScore(0, index_A, frame_id);

      // if (ground_truth_label == 0) {
      //   gtruth_label = Settings_->GroundTruthScoreWeak;
      // } else {
      //   gtruth_label = Settings_->GroundTruthScoreStrong;
      // }

      // ExponentialScore_AJ = multiplicity * (score_AJ - groundtruth_AJ), with score_AJ = Inner product between psi_Ji and U_Ai, over i.
      *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)) = score;
    }
    for (int &strong_label_valid : *CurrentValidIndices_Strong) {
      index_A      = GroundTruthLabelsTrainValStrong_[strong_label_valid].index_A;
      frame_id       = GroundTruthLabelsTrainValStrong_[strong_label_valid].frame_id;
      ground_truth_label = GroundTruthLabelsTrainValStrong_[strong_label_valid].ground_truth_label;

      score = this->ComputeScore(0, index_A, frame_id);

      // if (ground_truth_label == 0) {
      //   gtruth_label = Settings_->GroundTruthScoreWeak;
      // } else {
      //   gtruth_label = Settings_->GroundTruthScoreStrong;
      // }
      // ExponentialScore_AJ = multiplicity * (score_AJ - groundtruth_AJ), with score_AJ = Inner product between psi_Ji and U_Ai, over i.
      *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)) = score;
    }
  }

  if (type == 8) {
    // for (int weak_label_test; weak_label_test < Settings_->NumberOfWeakLabels_Test; weak_label_test++) {

    //   index_A      = GroundTruthLabelsTestWeak_[weak_label_test].index_A;
    //   frame_id       = GroundTruthLabelsTestWeak_[weak_label_test].frame_id;
    //   ground_truth_label = GroundTruthLabelsTestWeak_[weak_label_test].ground_truth_label;

    //   score = this->ComputeScore(0, index_A, frame_id);

    //   // if (ground_truth_label == 0) {
    //   //   gtruth_label = Settings_->GroundTruthScoreWeak;
    //   // } else {
    //   //   gtruth_label = Settings_->GroundTruthScoreStrong;
    //   // }
    //   // ExponentialScore_AJ = multiplicity * (score_AJ - groundtruth_AJ), with score_AJ = Inner product between psi_Ji and U_Ai, over i.
    //   *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod  )) = score;
    // }

    // for (int strong_label_test; strong_label_test < Settings_->NumberOfStrongLabels_Test; strong_label_test++) {
    //   index_A      = GroundTruthLabelsTestStrong_[strong_label_test].index_A;
    //   frame_id       = GroundTruthLabelsTestStrong_[strong_label_test].frame_id;
    //   ground_truth_label = GroundTruthLabelsTestStrong_[strong_label_test].ground_truth_label;

    //   score = this->ComputeScore(0, index_A, frame_id);

    //   // if (ground_truth_label == 0) {
    //   //   gtruth_label = Settings_->GroundTruthScoreWeak;
    //   // } else {
    //   //   gtruth_label = Settings_->GroundTruthScoreStrong;
    //   // }
    //   // ExponentialScore_AJ = multiplicity * (score_AJ - groundtruth_AJ), with score_AJ = Inner product between psi_Ji and U_Ai, over i.
    //   *(ConditionalScore->att(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)) = score;
    // }
  }
}

// Sparsity levels counter
int Trainer::countElementsAboveThreshold(MatrixBlob * blob, int col_lo, int col_hi, float threshold) {
  int n_abovethreshold = 0;
  float value = 0.;
  for (int row = 0; row < blob->rows; ++row)
  {
    for (int col = col_lo; col < col_hi; ++col)
    {
      value = blob->at(row, col);
      if (abs(value) > threshold) n_abovethreshold++;
    }
  }
  return n_abovethreshold;
}

float Trainer::getSparsityLevel(MatrixBlob * blob, int col_lo, int col_hi, float threshold) {
  assert(col_hi >= col_lo);
  if (col_hi == col_lo) {
    return 0.0;
  }
  else {
    int n_nonzeros = this->countElementsAboveThreshold(blob, col_lo, col_hi, threshold);
    return n_nonzeros / static_cast<float>(blob->rows * (col_hi - col_lo));
  }
}

float Trainer::getRunningAverage(VectorBlob * blob) {
  float running_average = 1.e10;
  for (int i = 0; i < std::min(blob->columns, Settings_->SparsityRunningAverageWindowSize); ++i)
  {
    running_average = (i*running_average + blob->at(blob->columns - 1 - i)) / static_cast<float>(i + 1);
  }
  return running_average;
}

float Trainer::getGradient(VectorBlob * blob, int index) {
  if (index > 0) {
    return blob->at(index) - blob->at(index - 1);
  } else return 0.;
}
void Trainer::printDecision(MatrixBlob * blob, int col_lo, int col_hi, float sparsity_level, float running_average, float gradient_curr, float gradient_prev, int exit_code, string message) {
  Settings *Settings_   = this->Settings_;
  int total = (col_hi - col_lo) * blob->rows;
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Columns " + to_string(col_lo) + ":" + to_string(col_hi) + " -- #nonzero weights: " + to_string(static_cast<int>(sparsity_level * total))  + " / " + to_string(total) + " = " + to_string(col_hi - col_lo) + " * "  + to_string(blob->rows));
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Sp curr:   " + to_string(sparsity_level * 100) + " %");
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Sp runn avg: " + to_string(running_average * 100) + " %");
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Sp desired:  " + to_string(Settings_->DesiredSparsityLevel * 100) + " %");
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Sp gradient: " + to_string(gradient_curr * 100) + " %");
  PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Decision:  " + message + " | Exit-code: " + to_string(exit_code));
  cout << endl;
}
void Trainer::checkSparsity(MatrixBlob *blob, VectorBlob * sp_lvls, VectorBlob * sp_nonzeros, int col_lo, int col_hi) {
  Settings *Settings_   = this->Settings_;
  float sparsity_level  = 1.;

  PrintFancy(Settings_->session_start_time, "Checking sparsity of " + blob->name);

  // Compute sparsity level
  sparsity_level = this->getSparsityLevel(blob, col_lo, col_hi, Settings_->SPARSITY_THRESHOLD);
  assert (sparsity_level >= 0. and sparsity_level <= 1.);

  // Store results
  sp_lvls->push_back(sparsity_level);
  sp_nonzeros->push_back(sparsity_level * (col_hi - col_lo) * blob->rows);
}
int Trainer::tuneSparsityCore(MatrixBlob * blob, VectorBlob * sp_lvls, VectorBlob * sp_nonzeros, int col_lo, int col_hi){
  Settings *Settings_   = this->Settings_;
  float sparsity_level  = 1.;
  float running_average = 0.;
  float gradient_curr   = 0;
  float gradient_prev   = 0;
  int exit_code     = -1;
  string message    = "";

  // Compute sparsity level
  sparsity_level = this->getSparsityLevel(blob, col_lo, col_hi, Settings_->SPARSITY_THRESHOLD);
  assert (sparsity_level >= 0. and sparsity_level <= 1.);

  // Store results
  sp_lvls->push_back(sparsity_level);
  sp_nonzeros->push_back(sparsity_level * (col_hi - col_lo) * blob->rows);

  // Check the running average and decide to lower regularization or not
  running_average = this->getRunningAverage(sp_lvls);

  // cout << "Check: sp_lvls->columns: " << sp_lvls->columns << " window-size: " << Settings_->SparsityRunningAverageWindowSize << endl;

  gradient_curr = this->getGradient(sp_lvls, sp_lvls->columns - 1);
  // gradient_prev = this->getGradient(sp_lvls, sp_lvls->columns - 2);

  if (sparsity_level < Settings_->DesiredSparsityLevel) {
    if (running_average > Settings_->DesiredSparsityLevel) {
      if (gradient_curr > 0.0) {
        message   = "Gradient increasing =( --> Lower regularization and continue";
        exit_code = 2;
      } else {
        message   = "Gradient is not increasing, so continue with current regularization strength";
        exit_code = 1;
      }
    } else {
      if (sp_lvls->columns < Settings_->SparsityRunningAverageWindowSize){
        message = "Sparsity and running avg below threshold, but we have not seen enough sparsity checks yet --> lower regularization strength and try again";
        exit_code = 3;
      } else {
        message = "Sparsity has stabilized + sparsity + running avg under threshold --> exit training";
        exit_code = 0;
      }
    }
  } else {
    if (running_average > Settings_->DesiredSparsityLevel) {
      message = "Continue with current regularization strength";
      exit_code = 1;
    } else {
      if (sp_lvls->columns < Settings_->SparsityRunningAverageWindowSize){
        message = "Running avg below threshold, but we have not seen enough sparsity checks yet --> lower regularization strength and try again";
        exit_code = 3; // We have not seen enough sparsity checks yet --> lower regularization strength and try again
      } else {
        message = "Sparsity has stabilized + sparsity > threshold BUT running avg < threshold --> still exit training";
        exit_code = 0;
      }
    }
  }

  printDecision(blob, col_lo, col_hi, sparsity_level, running_average, gradient_curr, gradient_prev, exit_code, message);

  // PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] Sparsity levels: ");
  // sp_lvls->showVectorContents(1000, 10);
  // PrintFancy(Settings_->session_start_time, "[SparsityTuning " + blob->name + "] # Non-zeros:   ");
  // sp_nonzeros->showVectorContents(1000, 10);
  // cout << endl;

  return exit_code;
}
int Trainer::tuneSparsity(MatrixBlob * blob, int subdim){

  Settings *Settings_  = this->Settings_;
  float sparsity_level_1 = 1., sparsity_level_2 = 1.;
  int exit_code      = -1;
  float running_average  = 0.;

  if ((subdim == 1) and (Settings_->StageThree_TrainSubDimension_1 == 1) and (Settings_->StageThree_RegularizationType_1 == 1 or Settings_->StageThree_RegularizationType_1 == 3)) {
    exit_code = this->tuneSparsityCore(blob, &blob->sp_sd1_lvls, &blob->sp_sd1_nonzeros, 0, Settings_->StageThree_SubDimension_1);
  }

  if ((subdim == 2) and (Settings_->StageThree_TrainSubDimension_2 == 1) and (Settings_->StageThree_RegularizationType_2 == 1 or Settings_->StageThree_RegularizationType_2 == 3)) {
    exit_code = this->tuneSparsityCore(blob, &blob->sp_sd2_lvls, &blob->sp_sd2_nonzeros, Settings_->StageThree_SubDimension_1, Settings_->StageThree_SubDimension_1 + Settings_->StageThree_SubDimension_2);
  }

  return exit_code;
}
void Trainer::tuneSparsityLevel(int batch, MatrixBlob * blob, int factor, int subdim) {

  float old_reg = 0.;

  if (factor == 1) {
    if (subdim == 1) {
      old_reg = Settings_->StageThree_A_RegularizationStrength_1_Sparse;
      Settings_->StageThree_A_RegularizationStrength_1_Sparse *= TUNE_SPARSITY_FACTOR;
    }
    if (subdim == 2) {
      old_reg = Settings_->StageThree_A_RegularizationStrength_2_Sparse;
      Settings_->StageThree_A_RegularizationStrength_2_Sparse *= TUNE_SPARSITY_FACTOR;
    }
  }
  if (factor == 2) {
    if (subdim == 1) {
      old_reg = Settings_->StageThree_B_RegularizationStrength_1_Sparse;
      Settings_->StageThree_B_RegularizationStrength_1_Sparse *= TUNE_SPARSITY_FACTOR;
    }
    if (subdim == 2) {
      old_reg = Settings_->StageThree_B_RegularizationStrength_2_Sparse;
      Settings_->StageThree_B_RegularizationStrength_2_Sparse *= TUNE_SPARSITY_FACTOR;
    }
  }
  if (factor == 3) {
    if (subdim == 1) {
      old_reg = Settings_->StageThree_C_RegularizationStrength_1_Sparse;
      Settings_->StageThree_C_RegularizationStrength_1_Sparse *= TUNE_SPARSITY_FACTOR;
    }
    if (subdim == 2) {
      old_reg = Settings_->StageThree_C_RegularizationStrength_2_Sparse;
      Settings_->StageThree_C_RegularizationStrength_2_Sparse *= TUNE_SPARSITY_FACTOR;
    }
  }

  cout << " [SparsityTuning " << blob->name << "] Tuning: ";
  cout << old_reg << " --> " << old_reg * TUNE_SPARSITY_FACTOR << " -- tuning factor: " << TUNE_SPARSITY_FACTOR << endl;
}
int Trainer::tuneSparsityAll(int batch, MatrixBlob * LF_A, MatrixBlob * LF_B, MatrixBlob * LF_C) {

  Settings *Settings_ = this->Settings_;
  int *exit_code    = new int[6];
  for (int i = 0; i < 6; ++i) exit_code[i] = -1;

  PrintFancy(Settings_->session_start_time, "[SparsityTuning] Batch: " + to_string(batch));

  if ((batch - Settings_->TrainFrequencyOffset_A) % Settings_->StageThree_Train_LF_A_EveryBatch == 0 and batch > 0) {
    exit_code[0] = this->tuneSparsity( LF_A, 1 );
    exit_code[1] = this->tuneSparsity( LF_A, 2 );
  }
  if ((batch - Settings_->TrainFrequencyOffset_B) % Settings_->StageThree_Train_LF_B_EveryBatch == 0 and batch > 0) {
    exit_code[2] = this->tuneSparsity( LF_B, 1 );
    exit_code[3] = this->tuneSparsity( LF_B, 2 );
  }
  if ((batch - Settings_->TrainFrequencyOffset_C) % Settings_->StageThree_Train_LF_C_EveryBatch == 0 and batch > 0) {
    exit_code[4] = this->tuneSparsity( LF_C, 1 );
    exit_code[5] = this->tuneSparsity( LF_C, 2 );
  }

  // Trick: if nothing has been trained before we check sparsity, we put one exit-code to 1
  if ((batch - Settings_->TrainFrequencyOffset_A) < Settings_->StageThree_Train_LF_A_EveryBatch and (batch - Settings_->TrainFrequencyOffset_B) < Settings_->StageThree_Train_LF_B_EveryBatch and (batch - Settings_->TrainFrequencyOffset_C) < Settings_->StageThree_Train_LF_C_EveryBatch) {
    exit_code[0] = 1;
  }

  // If needed tune regularization strengths
  for (int i = 0; i < 6; ++i) {
    if (exit_code[i] == 2 or exit_code[i] == 3) {
      if (i == 0) this->tuneSparsityLevel(batch, LF_A, 1, 1);
      if (i == 1) this->tuneSparsityLevel(batch, LF_A, 1, 2);
      if (i == 2) this->tuneSparsityLevel(batch, LF_B, 2, 1);
      if (i == 3) this->tuneSparsityLevel(batch, LF_B, 2, 2);
      if (i == 4) this->tuneSparsityLevel(batch, LF_C, 3, 1);
      if (i == 5) this->tuneSparsityLevel(batch, LF_C, 3, 2);
    }
  }

  // Store current sparsitylevels for snapshot title
  Settings_->CurrentSparsityLevel_B_SubDim1 = LF_B->sp_sd1_lvls.last();
  Settings_->CurrentSparsityLevel_B_SubDim2 = LF_B->sp_sd2_lvls.last();
  Settings_->CurrentSparsityLevel_C_SubDim1 = LF_C->sp_sd1_lvls.last();
  Settings_->CurrentSparsityLevel_C_SubDim2 = LF_C->sp_sd2_lvls.last();

  PrintDelimiter(0, 1, 80, '=');
  PrintFancy(Settings_->session_start_time, "Exit codes received:");
  for (int i = 0; i < 6; ++i) cout << exit_code[i] << " ";
  PrintDelimiter(1, 0, 80, '=');

  return *std::max_element(exit_code, exit_code + 6);
}


float Trainer::boundScore(int frame_id, float score) {

  Settings *Settings_ = this->Settings_;

  if (score > LOSS_SCORE_BLOWUP_THRESHOLD_ABOVE) {

    if (Settings_->EnableDebugPrinter_Level2 == 1 and frame_id % 100000 == 0) {
      cout << "Blow-up! F " << frame_id << " S: " << score << endl;
    }

    return LOSS_SCORE_BLOWUP_THRESHOLD_ABOVE;
  }
  else if (score < LOSS_SCORE_BLOWUP_THRESHOLD_BELOW) {

    if (Settings_->EnableDebugPrinter_Level2 == 1 and frame_id % 100000 == 0) {
      cout << "Blow-down! F " << frame_id << " S: " << score << endl;
    }

    return LOSS_SCORE_BLOWUP_THRESHOLD_BELOW;
  }
  else {
    return score;
  }
}
float Trainer::ComputeLossCore(std::vector<int> * WeakIndices, std::vector<int> * StrongIndices) {

  MatrixBlob     *ConditionalScore         = &(this->ConditionalScore);
  Settings     *Settings_            = this->Settings_;
  CurrentStateBlob *CurrentStateBlob         = this->CurrentStateBlob_;
  GroundTruthLabel *GroundTruthLabelsTrainValWeak_   = this->GroundTruthLabelsTrainValWeak_;
  GroundTruthLabel *GroundTruthLabelsTrainValStrong_ = this->GroundTruthLabelsTrainValStrong_;

  float loss    = 0.;
  float total   = 0.;
  float total_weak= 0.;
  float coeff_old = 0.;
  float coeff_new = 0.;
  float score   = 0.;
  int index_A   = 0;
  int frame_id  = 0;
  int n_used    = 0;
  int n_weak    = 0;

  for (int &weak_index : *WeakIndices) {

    index_A  = GroundTruthLabelsTrainValWeak_[weak_index].index_A;
    frame_id = GroundTruthLabelsTrainValWeak_[weak_index].frame_id;

    if (isnan(ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod))) {
      PrintFancy(Settings_->session_start_time, "ConditionalScore NaN! - " + to_string(frame_id) + " " + to_string(ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)));
    } else {

      total   += Settings_->TaskWeight.at(index_A) * Settings_->LossWeightWeak;
      coeff_new = Settings_->TaskWeight.at(index_A) * Settings_->LossWeightWeak / total;
      coeff_old = 1. - coeff_new;

      score   = this->boundScore(frame_id, ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod));

      loss    = coeff_old * loss + coeff_new * log( 1.0 + exp( score ) );

      n_used += 1;

      if (Settings_->EnableDebugPrinter_Level2 == 1 and n_used % 10000 == 0) {
        printf("x %10.8i f %10.8i task %4.2i tweight %5.3f || -1 score %9.4f loss %9.6f rloss %9.6f weak \n", n_used, frame_id, index_A, Settings_->TaskWeight.at(index_A), score, log( 1.0 + exp(score)), loss);
      }
    }
  }

  total_weak = total;
  n_weak = n_used;
  n_used = 0;

  for (int &strong_index : *StrongIndices) {
    index_A  = GroundTruthLabelsTrainValStrong_[strong_index].index_A;
    frame_id = GroundTruthLabelsTrainValStrong_[strong_index].frame_id;
    if (Settings_->EnableDebugPrinter_Level1 == 1 and isnan(ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod))) {
      PrintFancy(Settings_->session_start_time, "ConditionalScore NaN! - " + to_string(frame_id) + " " + to_string(ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod)));
    } else {

      total   += Settings_->TaskWeight.at(index_A) * Settings_->LossWeightStrong;
      coeff_new = Settings_->TaskWeight.at(index_A) * Settings_->LossWeightStrong / total;
      coeff_old = 1. - coeff_new;

      score   = this->boundScore(frame_id, ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod));

      loss    = coeff_old * loss + coeff_new * log( 1.0 + exp( -score ) );

      n_used += 1;

      if (Settings_->EnableDebugPrinter_Level2 == 1 and n_used % 10000 == 0)
        printf("x %10.8i f %10.8i task %4.2i tweight %5.3f || +1 score %9.4f loss %9.6f rloss %9.6f strong \n", n_used + n_used, frame_id, index_A, Settings_->TaskWeight.at(index_A), score, log( 1.0 + exp(-score)), loss);
    }
  }
  return loss;
}

// New!
float Trainer::ComputeLoss(int type) {
  Settings     *Settings_             = this->Settings_;
  std::vector<int> *CurrentTrainIndices_Strong_Sorted = &this->CurrentTrainIndices_Strong_Sorted;
  std::vector<int> *CurrentTrainIndices_Weak_Sorted   = &this->CurrentTrainIndices_Weak_Sorted;

  std::vector<int> *CurrentValidIndices_Strong    = &this->CurrentValidIndices_Strong;
  std::vector<int> *CurrentValidIndices_Weak      = &this->CurrentValidIndices_Weak;

  assert (type == 6 or type == 7 or type == 8);

  if (type == 6) {
    return ComputeLossCore(CurrentTrainIndices_Weak_Sorted, CurrentTrainIndices_Strong_Sorted);
  }
  if (type == 7) {
    return ComputeLossCore(CurrentValidIndices_Weak, CurrentValidIndices_Strong);
  }
  if (type == 8) {
    return -1.0;
  }
}

void Trainer::ProcessComputedLosses(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test) {
  this->Results_LossTrain.push_back(loss_train);
  this->Results_LossValid.push_back(loss_valid);
  this->Results_LossTest.push_back(loss_test);
}
float Trainer::GetLassoGradient(int thread_id, float x){

  assert (!isnan(x) and !isinf(x));

  if (x > LASSO_ZERO_BAND)    return 1.0;
  else if (x < LASSO_ZERO_BAND) return -1.0;
  else              return GLOBAL_GRAD_AT_ZERO;

  return 0.0;
}
float Trainer::SoftThresholdUpdate(int thread_id, int sign, float old_value, float update_value){
  // Note that in this function, we assume gradient descent -- update_value is subtracted!
  Settings *Settings_ = this->Settings_;

  float trunc_range = Settings_->TruncatedLasso_Window;

  assert (!isnan(old_value) and !isinf(old_value) );
  assert (!isnan(update_value) and !isinf(update_value) );

  if (thread_id == 0 and update_value > 10.0) {
    cout << "SoftThresholdUpdate: old: " << old_value << " delta " << update_value << endl;
  }

  // Truncated soft thresholding: if weight too small, don't apply softthreshold update, otherwise, we do.
  if ( abs(old_value) < trunc_range ) {
    return old_value;
  } else {
    if (old_value == 0.0) {
      return 0.0;
    }
    else if (old_value > 0. and old_value + sign * update_value <= 0.) {
      // if (thread_id == 0) cout << "Debug SoftThresholdUpdate: " << std::setw(15) << std::setprecision(10) << old_value << " " << update_value << " " << old_value + sign * update_value << " return: " << 0. << endl;
      return 0.;
    }
    else if (old_value < 0. and old_value + sign * update_value >= 0.) {
      // if (thread_id == 0) cout << "Debug SoftThresholdUpdate: " << std::setw(15) << std::setprecision(10) << old_value << " " << update_value << " " << old_value + sign * update_value << " return: " << 0. << endl;
      return 0.;
    }
    else {
      // if (thread_id == 0) cout << "Debug SoftThresholdUpdate: " << std::setw(15) << std::setprecision(10) << old_value << " " << update_value << " " << old_value + sign * update_value << " return: " << old_value + sign * update_value << endl;
      return old_value + sign * update_value;
    }
  }
}
void Trainer::UpdateTruncatedLassoWindow(int batch){
  Settings *Settings_ = this->Settings_;

  Settings_->TruncatedLasso_Window *= Settings_->TruncatedLasso_Window_ScaleFactor;

  PrintFancy(Settings_->session_start_time, "Scaling Truncated Lasso Window down");
  printf("Truncated LASSO window -- %.10f --> %.10f (%4.2fx) -- updating every %i batches \n",  Settings_->TruncatedLasso_Window,
        Settings_->TruncatedLasso_Window_ScaleFactor * Settings_->TruncatedLasso_Window,Settings_->TruncatedLasso_Window_ScaleFactor,
        Settings_->TruncatedLasso_Window_ScaleFactor,
        Settings_->TruncatedLasso_Window_UpdateFrequency);
}

FILE * Trainer::LoadFile(string fn, int expected_floats_in_file, int n_seek_float) {
  FILE * fp;
  fp = fopen(fn.c_str() , "rb");

  if (fp == NULL) {
    fputs("Error!", stderr);
    PrintFancy(Settings_->session_start_time, "Could not read " + fn);
    exit(1);
  }
  int result = 0;
  float * float_ = new float;
  int n_floats_in_file = 0;
  while ((result = fread(float_, sizeof(*float_), 1, fp))) n_floats_in_file++;
  PrintFancy(Settings_->session_start_time, "File: " + fn + " -- read n_floats_in_file: " + to_string(n_floats_in_file));

  if (n_floats_in_file != expected_floats_in_file){
    cout << "n_floats_in_file     : " << n_floats_in_file << endl;
    cout << "expected_floats_in_file: " << expected_floats_in_file << endl;
    assert(n_floats_in_file == expected_floats_in_file);
  }

  fseek(fp, 0, SEEK_SET);
  fseek(fp, n_seek_float * sizeof(float), SEEK_SET);
  cout << endl;
  cout << "Seeking to position " << n_seek_float << " / " << n_floats_in_file << " in " << fn << endl;
  cout << "This is " << n_seek_float * sizeof(float) << " bytes" << endl;

  delete float_;
  return fp;
}
void Trainer::LoadSnapshotWeight( Tensor3Blob * blob, string fn, int size_of_container, int start_of_container, int slice_start, int slice_end, int row_start, int row_end, int col_start, int col_end ) {
  Settings *Settings_   = this->Settings_;
  PrintFancy(Settings_->session_start_time, "Loading weights from snapshot for blob " + blob->name );
  FILE * fp = this->LoadFile(fn.c_str(), size_of_container, start_of_container);
  float   *float_ = new float;
  for (int k = slice_start; k < slice_end; ++k) {
    for (int i = row_start; i < row_end; ++i) {
      for (int j = col_start; j < col_end; ++j) {
        fread(float_, sizeof(*float_), 1, fp);
        *(blob->att(i, j, k)) = *float_;
      }
    }
  }
  delete float_;
  fclose(fp);
}
void Trainer::LoadSnapshotWeight( MatrixBlob * blob, string fn, int size_of_container, int start_of_container, int row_start, int row_end, int col_start, int col_end ) {
  Settings *Settings_   = this->Settings_;
  PrintFancy(Settings_->session_start_time, "Loading weights from snapshot for blob " + blob->name );
  FILE * fp = this->LoadFile(fn.c_str(), size_of_container, start_of_container);
  float   *float_ = new float;
  for (int i = row_start; i < row_end; ++i) {
    for (int j = col_start; j < col_end; ++j) {
      fread(float_, sizeof(*float_), 1, fp);
      *(blob->att(i, j)) = *float_;
    }
  }
  delete float_;
  fclose(fp);
}
void Trainer::LoadSnapshotWeight( VectorBlob * blob, string fn, int size_of_container, int start_of_container, int col_start, int col_end ) {

  Settings *Settings_   = this->Settings_;
  PrintFancy(Settings_->session_start_time, "Loading weights from snapshot for blob " + blob->name );
  FILE * fp = this->LoadFile(fn.c_str(), size_of_container, start_of_container);
  float   *float_ = new float;

  for (int j = col_start; j < col_end; ++j) {

    fread(float_, sizeof(*float_), 1, fp);
    *(blob->att(j)) = *float_;
    if (Settings_->EnableDebugPrinter_Level3 == 1) {
      cout << "VectorBlob readin debug: col " << j << " float: " << *(blob->att(j)) << " " << *float_ << endl;
    }
  }
  delete float_;
  fclose(fp);
}
int Trainer::GetExpectedSizeSnapshotFile(Settings * s){
  Settings *Settings_   = this->Settings_;

  PrintFancy(Settings_->session_start_time, "Volatile!!! Bias must be loaded from a snapshot, but we interchangably use snapshots from S1, S3 LR-only and S3 L+S. Fix this in the future.");

  // Use when loading from a L+S snapshot
  if (Settings_->DATASET_NAME == "bb") {
    PrintFancy(Settings_->session_start_time, "Basketball: use low-res L+S snapshot with latent dim 10 (at this time, has the best performance)");
    return (Settings_->Dimension_A + Settings_->StageOne_Dimension_B + Settings_->StageOne_Dimension_C) * 10 \
        + Settings_->Dimension_A * Settings_->StageOne_Dimension_B * Settings_->StageOne_Dimension_C \
        + Settings_->Dimension_A;
  }
  if (Settings_->DATASET_NAME == "fvf") {
    // PrintFancy(Settings_->session_start_time, "FvF: use L+S snapshot with latent dim 3 (at this time, has the best performance)");
    PrintFancy(Settings_->session_start_time, "FvF: for stage 5, use L+LS snapshot with latent dim 20 (at this time, has the best performance)");
    return (Settings_->Dimension_A + Settings_->StageThree_Dimension_B + Settings_->StageThree_Dimension_C) * 20 \
        + Settings_->Dimension_A * Settings_->StageThree_Dimension_B * Settings_->StageThree_Dimension_C * 0 \
        + Settings_->Dimension_A;
  }
  // return Settings_->Dimension_A * Settings_->StageOne_Dimension_B * Settings_->StageOne_Dimension_C + Settings_->Dimension_A;
}
float Trainer::cap(int thread_id, int index_A, int frame_id, float score, float cap_hi, float cap_lo) {
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

void Trainer::ComputeSpatialEntropy(int thread_id,
  int index_A,
  int frame_id,
  int ground_truth_label,
  int n_dimension_B, //           = Settings_->StageOne_Dimension_B;
  int n_dimension_C, //          = Settings_->StageOne_Dimension_C;
  float dresponsefunction_dB,
  int MiniBatchSize // = Settings_->StageOne_MiniBatchSize
  ) {
  MatrixBlob *ConditionalScore = &(this->ConditionalScore);
  float strongweakweight       = 0.;
  float dP_df                  = 0.;
  float U_gradient             = 0.;
  float ExponentialScore       = ConditionalScore->at(frame_id, index_A % Settings_->ScoreFunctionColumnIndexMod);
  float df_dB                  = dresponsefunction_dB;

  if (ground_truth_label == 0) {
    strongweakweight = Settings_->LossWeightWeak;
    dP_df = 1.0 / ( 1.0 + exp(-ExponentialScore) );
  } else {
    assert(ground_truth_label == 1);
    strongweakweight = Settings_->LossWeightStrong;
    dP_df = -1.0 / ( 1.0 + exp(ExponentialScore) );
  }

  std::vector<int> indices_B = this->getIndices_B(frame_id, 1);
  std::vector<int> indices_C = this->getIndices_C(frame_id, 1);

  U_gradient = strongweakweight * dP_df * df_dB / MiniBatchSize;

  // Process all positions in this feature vector
  // Watch out: our position features have a stop token in the last slot, which we should *NOT* process!
  for (int i = 0; i < indices_B.size() - 1; i++) {
    int index_B = indices_B[i];
    spatial_entropy.AddGradientToHistogram(index_B, U_gradient);
  }
}