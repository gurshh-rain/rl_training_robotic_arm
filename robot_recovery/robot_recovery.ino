#include <ESP32Servo.h>

Servo mg995;
const int servoPin = 14; // Safe PWM-capable GPIO pin

void setup() {
  mg995.setPeriodHertz(50);          // Standard 50Hz servo signal
  mg995.attach(servoPin, 1000, 2000); // Typical MG995 pulse range (us)
  mg995.write(90);                    // Move to 90 degrees
}

void loop() {
  mg995.write(0);
  delay(1000);

  mg995.write(90);
  delay(1000);

  mg995.write(180);
  delay(1000);
}
