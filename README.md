# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim:

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required:

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm:

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:
```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

withglass = cv2.imread('image_02.png', 0)
group = cv2.imread('image_03.png', 0)


plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")
plt.show()

plt.imshow(group, cmap='gray')
plt.title("Group Image")
plt.show()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    return face_img

def detect_eyes(img):
    eye_img = img.copy()
    eyes = eye_cascade.detectMultiScale(eye_img)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    return eye_img

result_withglass_faces = detect_face(withglass)
plt.imshow(result_withglass_faces, cmap='gray')
plt.title("Faces in With Glasses Image")
plt.show()

result_group_faces = detect_face(group)
plt.imshow(result_group_faces, cmap='gray')
plt.title("Faces in Group Image")
plt.show()

result_withglass_eyes = detect_eyes(withglass)
plt.imshow(result_withglass_eyes, cmap='gray')
plt.title("Eyes in With Glasses Image")
plt.show()

result_group_eyes = detect_eyes(group)
plt.imshow(result_group_eyes, cmap='gray')
plt.title("Eyes in Group Image")
plt.show()
```

## Output:
<img width="638" height="593" alt="Screenshot 2025-11-16 185724" src="https://github.com/user-attachments/assets/c957af9d-08b2-4f31-b217-dbc784c94ffb" />
<img width="805" height="510" alt="Screenshot 2025-11-16 185737" src="https://github.com/user-attachments/assets/3c968173-4b1c-4de6-94a2-de4cd59f9819" />
<img width="639" height="595" alt="Screenshot 2025-11-16 185750" src="https://github.com/user-attachments/assets/13c30560-b403-404a-8851-8288d87d17be" />
<img width="784" height="514" alt="Screenshot 2025-11-16 185802" src="https://github.com/user-attachments/assets/94f9dedd-4e2f-45f6-ab7e-d02bb8c44848" />
<img width="619" height="594" alt="Screenshot 2025-11-16 185811" src="https://github.com/user-attachments/assets/7584ea98-730d-45bf-9d42-6bf6ea497573" />
<img width="769" height="509" alt="Screenshot 2025-11-16 185820" src="https://github.com/user-attachments/assets/12a3c7d8-e683-4f31-99e1-af111f28ca36" />

## Result:
Thus, the Python program for Face Detection using Haar Cascades with OpenCV and Matplotlib is implemented and executed successfully.



