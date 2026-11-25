#pragma once

#include <math.h>

// World up, expressed in the rotating base frame. Accelerometer direction is
// an absolute reference at rest but includes linear acceleration during push-off.
struct UpEstimate {
  float up[3] = {};
  bool initialized = false;

  void update(const float accelUp[3], const float gyro[3], float dt) {
    if (!initialized) {
      for (int i = 0; i < 3; ++i) up[i] = accelUp[i];
      initialized = true;
      return;
    }
    float next[3];
    float norm2 = 0;
    const float gain = dt / (0.35f + dt);
    for (int i = 0; i < 3; ++i) {
      int j = (i + 1) % 3, k = (i + 2) % 3;
      float predicted = up[i] + dt * (up[j] * gyro[k] - up[k] * gyro[j]);
      next[i] = predicted + gain * (accelUp[i] - predicted);
      norm2 += next[i] * next[i];
    }
    float norm = sqrtf(norm2);
    for (int i = 0; i < 3; ++i) up[i] = next[i] / norm;
  }
};

// Episode history is distinct from instantaneous pose: a slow fall passes
// through "not flat" before it crosses the threshold for a new fall.
struct FallWatch {
  bool standing = false;
  bool hasStood = false;
  void begin(bool uprightStart) { standing = false; hasStood = uprightStart; }
  void confirmStanding() { standing = hasStood = true; }
  void leaveStanding() { standing = false; }
  bool newFall(float upY) const { return hasStood && upY > -0.75f; }
};
