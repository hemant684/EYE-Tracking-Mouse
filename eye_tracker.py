import tkinter as tk
from tkinter import ttk
import cv2
import dlib
import numpy as np
import pyautogui as pag
from PIL import Image, ImageTk
import threading
import time
import os
from collections import deque

class EnhancedEyeTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Eye Mouse Tracker")
        pag.FAILSAFE = False
        
        # Tracking variables
        self.is_tracking = False
        self.screen_w, self.screen_h = pag.size()
        
        # Dlib face detection
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            raise FileNotFoundError("Missing shape predictor file!")
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Smoothing and cursor movement
        self.smoothing_window = 15  # Number of frames for smoothing
        self.position_history = deque(maxlen=self.smoothing_window)
        self.last_valid_position = None
        
        # Simplified Kalman filter (position only)
        self.kalman = cv2.KalmanFilter(2, 2)
        self.kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        self.setup_gui()

    def setup_gui(self):
        # GUI setup remains the same as original
        # ... (same as original code)
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, columnspan=2, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Controls frame
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")
        self.controls_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Buttons
        self.toggle_button = ttk.Button(self.controls_frame, text="Start Tracking", command=self.toggle_tracking)
        self.toggle_button.grid(row=0, column=0, padx=5)
        
        # Sliders
        self.blink_label = ttk.Label(self.controls_frame, text="Blink Threshold:")
        self.blink_label.grid(row=0, column=1, padx=5)
        self.blink_slider = ttk.Scale(
            self.controls_frame,
            from_=0.05,
            to=0.3,
            orient=tk.HORIZONTAL,
            value=0.15,
            command=self.update_blink_threshold
        )
        self.blink_slider.grid(row=0, column=2, padx=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
    def update_blink_threshold(self, value):
        self.blink_threshold = float(value)
        
    def eye_aspect_ratio(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR) for blink detection."""
        p2_p6 = abs(eye_points[1].y - eye_points[5].y)
        p3_p5 = abs(eye_points[2].y - eye_points[4].y)
        p1_p4 = abs(eye_points[0].x - eye_points[3].x)
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def smooth_position(self, new_pos):
        """Enhanced smoothing with Kalman filter and weighted moving average."""
        if new_pos is None:
            return self.last_valid_position  # Return last valid position if no new data
            
        # Kalman filter update
        measurement = np.array([[np.float32(new_pos[0])], [np.float32(new_pos[1])]])
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        kalman_pos = (int(prediction[0]), int(prediction[1]))
        
        # Weighted moving average
        self.position_history.append(kalman_pos)
        if len(self.position_history) >= self.smoothing_window:
            avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
            avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)
            smoothed_pos = (int(avg_x), int(avg_y))
        else:
            smoothed_pos = kalman_pos
        
        # Ensure cursor stays within screen bounds
        smoothed_pos = (
            max(0, min(self.screen_w, smoothed_pos[0])),
            max(0, min(self.screen_h, smoothed_pos[1]))
        )
        
        self.last_valid_position = smoothed_pos
        return smoothed_pos

    def tracking_loop(self):
        while self.is_tracking:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                # No face detected, skip this frame
                continue
                
            # Assume only one face is detected
            face = faces[0]
            landmarks = self.predictor(gray, face)
            
            # Get eye landmarks
            left_eye_points = [landmarks.part(i) for i in range(36, 42)]
            right_eye_points = [landmarks.part(i) for i in range(42, 48)]
            
            # Calculate eye centers
            left_eye_center = (
                sum(p.x for p in left_eye_points) // 6,
                sum(p.y for p in left_eye_points) // 6
            )
            right_eye_center = (
                sum(p.x for p in right_eye_points) // 6,
                sum(p.y for p in right_eye_points) // 6
            )
            
            # Use both eyes to calculate gaze point
            gaze_point = (
                (left_eye_center[0] + right_eye_center[0]) // 2,
                (left_eye_center[1] + right_eye_center[1]) // 2
            )
            
            # Map gaze point to screen coordinates
            screenx = int(np.interp(gaze_point[0], (0, frame.shape[1]), (0, self.screen_w)))
            screeny = int(np.interp(gaze_point[1], (0, frame.shape[0]), (0, self.screen_h)))
            
            # Apply smoothing
            smooth_pos = self.smooth_position((screenx, screeny))
            pag.moveTo(smooth_pos[0], smooth_pos[1], duration=0.05, _pause=False)
            
            # Visualization (optional)
            cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(frame, gaze_point, 5, (255, 0, 0), -1)
            
            # Update video feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.video_label.imgtk = None

    def toggle_tracking(self):
        if not self.is_tracking:
            self.is_tracking = True
            self.toggle_button.configure(text="Stop Tracking")
            self.status_var.set("Tracking active")
            # Reset Kalman filter state when starting tracking
            self.kalman.statePre = np.array([[0], [0]], np.float32)
            self.kalman.statePost = np.array([[0], [0]], np.float32)
            threading.Thread(target=self.tracking_loop, daemon=True).start()
        else:
            self.is_tracking = False
            self.toggle_button.configure(text="Start Tracking")
            self.status_var.set("Tracking stopped")

    def cleanup(self):
        self.is_tracking = False
        if self.cap.isOpened():
            self.cap.release()

def main():
    root = tk.Tk()
    app = EnhancedEyeTrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()