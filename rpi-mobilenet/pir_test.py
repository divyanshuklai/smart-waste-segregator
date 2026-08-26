import RPi.GPIO as GPIO
import time
from datetime import datetime

GPIO.setmode(GPIO.BCM)
PIR_PIN = 17
GPIO.setup(PIR_PIN, GPIO.IN)

print("PIR Motion Sensor Test (CTRL+C to exit)")
print("Waiting for sensor to settle...")
time.sleep(60)  # Give the sensor a full minute to calibrate
print("Ready to detect motion!")

try:
    previous_state = GPIO.input(PIR_PIN)
    while True:
        current_state = GPIO.input(PIR_PIN)
        if current_state != previous_state:
            if current_state:
                print(f"{datetime.now()}: Motion detected!")
            else:
                print(f"{datetime.now()}: Motion stopped")
            previous_state = current_state
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

except KeyboardInterrupt:
    print("\nTest ended by user")
    GPIO.cleanup()