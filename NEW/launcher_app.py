import tkinter as tk
from tkinter import ttk
import subprocess
import os
import signal

class ScriptLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Fish Counter Launcher")
        self.root.geometry("400x150")
        self.root.configure(bg="#2c3e50")

        self.process = None  # To keep track of the running script

        # --- UI Elements ---
        control_frame = tk.Frame(root, bg="#2c3e50", padx=20, pady=30)
        control_frame.pack(expand=True)

        self.start_button = ttk.Button(control_frame, text="🚀 Start Counting", command=self.start_script, width=20)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(control_frame, text="🛑 Stop Counting", command=self.stop_script, state=tk.DISABLED, width=20)
        self.stop_button.pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_script(self):
        if self.process is None:
            print("INFO: Starting program_counting_adapt.py...")
            # Command to run your script. Use 'python3' or 'python' as needed.
            command = ["python3", "program_counting_adapt.py"]
            
            # Popen starts the process without blocking the GUI
            self.process = subprocess.Popen(command)
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            print(f"INFO: Script started with Process ID: {self.process.pid}")

    def stop_script(self):
        if self.process:
            print(f"INFO: Stopping script with Process ID: {self.process.pid}...")
            # Terminate the process and all its children
            os.kill(self.process.pid, signal.SIGTERM)
            self.process = None
            
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            print("INFO: Script stopped.")

    def on_closing(self):
        """Ensure the script is stopped when the launcher window is closed."""
        self.stop_script()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptLauncher(root)
    root.mainloop()