#ifndef CURRENTSTATEBLOB_H
#define CURRENTSTATEBLOB_H

class CurrentStateBlob {
  public:
  string  session_id;
  std::chrono::high_resolution_clock::time_point   start_time;
  int   current_stage;
  int   current_epoch;
  int   current_cross_val_run;
  int   curr_snapshot_id;
  CurrentStateBlob() {
    curr_snapshot_id = 0;
  }
};

#endif