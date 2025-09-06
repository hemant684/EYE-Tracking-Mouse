from setuptools import setup

APP = ["eye_tracker.py"]  # Replace with your main script name
DATA_FILES = ["shape_predictor_68_face_landmarks.dat"]  # Include necessary files
OPTIONS = {
    "argv_emulation": True,
    "packages": ["tkinter", "cv2", "dlib", "numpy", "pyautogui", "PIL", "threading", 
                 "time", "os", "collections"],
    "includes": ["numpy", "cv2", "dlib", "pyautogui"],
    "resources": ["shape_predictor_68_face_landmarks.dat"],
    "iconfile": "app_icon.icns",  # macOS requires .icns format
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
