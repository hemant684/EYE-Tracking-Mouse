Eye Tracking Mouse Control System

This project implements an eye-tracking system that allows users to control their computer mouse cursor using eye movements. It's designed to assist people with limited mobility or those seeking alternative input methods.

Features
- Real-time eye tracking using computer vision
- Mouse cursor control through eye movements
- Blink detection for click actions
- Customizable sensitivity settings
- Low latency response time
- Cross-platform compatibility

Prerequisites
- Python 3.8+
- OpenCV
- dlib
- NumPy
- PyAutoGUI

Installation
1. Clone the repository:
```bash
git clone https://github.com/02falgun/Eye_Tracking_Mouse.git
cd Eye_Tracking_Mouse
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the shape predictor file:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Usage
1. Run the main script:
```bash
python eye_tracker.py
```

2. Position yourself in front of the webcam
3. Calibrate the system by following the on-screen instructions
4. Control the mouse cursor with your eye movements
   - Look in the direction you want to move the cursor
   - Blink to perform a click action

Configuration
Adjust settings in `config.py`:
- `SENSITIVITY`: Mouse movement sensitivity
- `SMOOTHING`: Cursor movement smoothing factor
- `CLICK_DURATION`: Blink duration for click detection

Troubleshooting
Common issues and solutions:

1. Camera not detected
   - Ensure your webcam is properly connected
   - Check camera permissions

2. Poor tracking accuracy
   - Adjust lighting conditions
   - Recalibrate the system
   - Update sensitivity settings

3. High CPU usage
   - Lower the frame processing resolution
   - Adjust the processing frequency

Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request


Acknowledgments
- dlib for facial landmark detection
- OpenCV community for computer vision tools
- PyAutoGUI for mouse control functionality

