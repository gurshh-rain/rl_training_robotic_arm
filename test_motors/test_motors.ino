#include <Arduino.h>
#include <ESP32Servo.h>

constexpr int servoPin = 25;  // Use 14 to test the upper servo instead.
Servo motor;

void setup() {
  // Runs once at power-up or reset. Remove the horn before testing.
  delay(3000);
  motor.setPeriodHertz(50);
  motor.attach(servoPin, 1000, 2000);
  motor.setTimerWidth(16);
  for (int angle = 0; angle <= 180; ++angle) {
    motor.write(angle);
    delay(20);
  }
  for (int angle = 180; angle >= 0; --angle) {
    motor.write(angle);
    delay(20);
  }
  motor.detach();
}

void loop() {}
