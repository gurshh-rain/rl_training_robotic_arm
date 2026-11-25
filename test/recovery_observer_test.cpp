#include "../robot_recovery/recovery_observer.h"
#include <cassert>

int main() {
  FallWatch falls;
  falls.begin(false);
  assert(!falls.newFall(0));  // First attempt, not a repeat fall.
  falls.confirmStanding();
  falls.leaveStanding();  // Left the flat band but not yet on another side.
  assert(!falls.standing && !falls.newFall(-0.88f));
  assert(falls.newFall(-0.70f));
  falls.begin(false);
  assert(!falls.newFall(-0.70f));

  UpEstimate filter;
  float gyro[] = {0, 0, 0}, flat[] = {0, -1, 0};
  float push[] = {0.6f, -0.8f, 0};
  filter.update(flat, gyro, 0.02f);
  filter.update(push, gyro, 0.02f);
  assert(fabsf(filter.up[0]) < 0.05f && filter.up[1] < -0.99f);

  filter.initialized = false;
  filter.update(flat, gyro, 0.02f);
  gyro[0] = 1.5707963f;
  for (int tick = 1; tick <= 50; ++tick) {
    float angle = gyro[0] * tick * 0.02f;
    float reference[] = {0, -cosf(angle), sinf(angle)};
    filter.update(reference, gyro, 0.02f);
    assert(fabsf(filter.up[2] - reference[2]) < 0.005f);
  }
}
