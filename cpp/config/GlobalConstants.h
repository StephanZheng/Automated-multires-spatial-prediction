#ifndef GLOBALCONSTANTS_H
#define GLOBALCONSTANTS_H

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <queue>

// ============================================================================
// Global constants
// ============================================================================

#define LEARNING_RATE_REDUCTION_FACTOR 0.9
#define NUMBER_OF_INTS_WINDOW_3BY3 9

// Layout stuff
#define PRINT_FLOAT_PRECISION 10
#define PRINT_FLOAT_PRECISION_LR 15
#define PRINT_FLOAT_PRECISION_LOSS 15
#define PRINT_FLOAT_PRECISION_SCORE 10
#define PRINT_DELIMITER_LENGTH 100

// Tweaks to LR update schedule
#define LR_SCHEDULE_VALID_TRAIN_UP_CHANGE_RATE_BY   0.5
#define LR_SCHEDULE_LOSS_TRAIN_UP_CHANGE_RATE_BY  0.5
#define LR_SCHEDULE_LOSS_TRAIN_DOWN_CHANGE_RATE_BY  1.25

// Blow-up detectors
#define SCORE_BLOWUP_THRESHOLD_BELOW -200.0
#define SCORE_BLOWUP_THRESHOLD_ABOVE 200.0

// Caps scores
#define LOSS_SCORE_BLOWUP_THRESHOLD_BELOW -50.0
#define LOSS_SCORE_BLOWUP_THRESHOLD_ABOVE 50.0

#define SCORE_BLOWUP_ADJUST_LR_BY_FACTOR 0.5
#define NAN_ENCOUNTERED -2
#define LOSS_BLOWUP_THRESHOLD 100.0
#define LOSS_INIT_VALUE 10000.0


// Sparse matrix regularization
#define SPARSE_MAT_S_ZERO_BAND 0.0
#define SPARSE_LF_B_ZERO_BAND 0.0
#define S1_WEIGHT_U_ZERO_BAND 0.0

#define ZERO_FLOAT 0.0

// Old out-of-bounds grid cell coordinate detectors
#define SKIP_OUT_OF_BOUNDS_INDICES_BH_S1 // if (index_B == 500) continue;
#define SKIP_OUT_OF_BOUNDS_INDICES_DEF_S1 // if (index_C == 36) continue;

#define SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S1 // if (indices_B->at(index_B) < 0) continue;
#define SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S1 // if (indices_C->at(index_C) < 0) continue;

#define SKIP_OUT_OF_BOUNDS_INDICES_BH_S3 // if (index_B == 2000 || index_B == 4001 || index_B == 6002) continue;
#define SKIP_OUT_OF_BOUNDS_INDICES_DEF_S3 // if (index_C == 144 || index_C == 289 || index_C == 434) continue;

#define SKIP_OUT_OF_BOUNDS_INDICES_BH_VEC_S3 // if (indices_B->at(index_B) < 0) continue;
#define SKIP_OUT_OF_BOUNDS_INDICES_DEF_VEC_S3 // if (indices_C->at(index_C) < 0) continue;

#define SENTINEL_OFFSET 1000000

#define HACK_SLIDINGWINDOW_DOUBLE_WEIGHTS 1

#define LASSO_ZERO_BAND 0.0
#define GLOBAL_GRAD_AT_ZERO 0.0

#define USE_SOFT_THRESHOLDING 1

#define NUMBER_OF_INTS_WINDOW_3BY3 9

#define TUNE_SPARSITY_FACTOR 0.5


#endif