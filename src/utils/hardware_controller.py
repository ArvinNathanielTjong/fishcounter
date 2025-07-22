# (To be added near the top of launcher_app.py)
import OPi.GPIO as GPIO
import time
import threading

class BuzzerController:
    def __init__(self, pin):
        self.pin = pin
        GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, GPIO.LOW) # Ensure buzzer is off initially

    def _beep_thread(self, duration=0.2):
        """The actual beep logic that runs in a separate thread."""
        GPIO.output(self.pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(self.pin, GPIO.LOW)

    def beep(self, duration=0.2):
        """Starts a non-blocking beep."""
        # Run the beep in a thread so it doesn't freeze the GUI
        thread = threading.Thread(target=self._beep_thread, args=(duration,), daemon=True)
        thread.start()

    def cleanup(self):
        """Cleans up GPIO resources."""
        GPIO.cleanup()