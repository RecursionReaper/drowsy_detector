# Sleep Detection using Eye Aspect Ratio (EAR)

This project implements a real-time sleep detection system using OpenCV, Dlib, and SciPy. The system detects facial landmarks and calculates the Eye Aspect Ratio (EAR) to determine whether a person's eyes are closed for an extended period, indicating possible drowsiness.

## Features
- Real-time face and eye detection using Dlib's facial landmark predictor.
- Calculation of the Eye Aspect Ratio (EAR) to detect eye closure.
- Visual indication when eyes are closed for a predefined number of frames.
- Uses OpenCV for video capture and display.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install opencv-python dlib numpy scipy imutils
```

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/RecursionReaper/drowsy_detector.git
   cd <repository_folder>
   ```

2. Place the `shape_predictor_68_face_landmarks.dat` file in the same directory as `sleep.py`. You can download it from:
   [Dlib's Pretrained Model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

3. Extract and ensure the `.dat` file is in the correct location.

## Running the Script
Activate your virtual environment (if applicable):

```bash
source sleep-detection-env/bin/activate
```

Run the script:

```bash
python sleep.py
```

## How It Works
- The script captures video from your webcam.
- It detects the face and eye landmarks using Dlib's shape predictor.
- The EAR is calculated for both eyes and averaged.
- If the EAR value falls below a threshold (default: 0.2) for a consecutive number of frames (default: 5), a warning is displayed.
- The program exits when you press the `q` key.

## Parameters
You can fine-tune the detection sensitivity by modifying these parameters in `sleep.py`:

```python
EAR_THRESHOLD = 0.2  # Adjust based on testing
CLOSED_FRAMES_THRESHOLD = 5  # Number of consecutive frames before triggering an alert
```

## Expected Output
- A real-time video feed with face and eye landmarks drawn.
- A text overlay displaying the EAR value.
- A warning message "EYES CLOSED!" when prolonged eye closure is detected.

## Troubleshooting
### Common Issues & Fixes

1. **Dlib Module Not Found**
   - Ensure Dlib is installed correctly:
     ```bash
     pip install dlib
     ```
   - Try reinstalling with:
     ```bash
     pip uninstall dlib && pip install dlib
     ```

2. **Shape Predictor File Not Found**
   - Ensure `shape_predictor_68_face_landmarks.dat` is in the correct directory.
   - Use the absolute path if necessary:
     ```python
     predictor = dlib.shape_predictor("/absolute/path/to/shape_predictor_68_face_landmarks.dat")
     ```

3. **Camera Not Working**
   - Ensure your webcam is properly connected.
   - Try changing the video capture index in `cap = cv2.VideoCapture(0)`. Replace `0` with `1` or `2` if necessary.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Dlib for facial landmark detection.
- OpenCV for image processing.
- SciPy for spatial distance calculations.
