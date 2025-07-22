# hardware_controller.py
from periphery import GPIO
import time
import threading

class BuzzerController:
    def __init__(self, gpio_num):
        self.gpio_num = gpio_num
        self.buzzer = GPIO(self.gpio_num, "out")
        self.buzzer.write(False)  # Ensure buzzer is off initially

    def _beep_thread(self, duration=0.2):
        """The actual beep logic that runs in a separate thread."""
        self.buzzer.write(True)
        time.sleep(duration)
        self.buzzer.write(False)

    def beep(self, duration=0.2):
        """Starts a non-blocking beep."""
        thread = threading.Thread(target=self._beep_thread, args=(duration,), daemon=True)
        thread.start()

    def cleanup(self):
        """Cleans up GPIO resources."""
        self.buzzer.write(False)
        self.buzzer.close()
