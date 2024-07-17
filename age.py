import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras_cv_attention_models import fastvit
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def preload_model(path):
    """ Load and Initialise the model """
    custom_objects = {
        'FastViT_T8': fastvit.FastViT_T8,
        'GlobalAveragePooling2D': GlobalAveragePooling2D,
        'Dense': Dense,
        'Dropout': Dropout
    }
    model = load_model(path, custom_objects=custom_objects, compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def pre_process_image(img):
    """ Pre-process the image for neural network """
    img = img.astype(np.float32)
    pp_image = cv2.resize(img, (256, 256))  # resize image
    pp_image = pp_image / 255.  # normalise image
    pp_image = np.expand_dims(pp_image, axis=0)  # add batch dimension

    return pp_image


def predict_age(img, model):
    """Predict Age"""
    image_pp = pre_process_image(img)
    preds = model.predict(image_pp)
    age = preds[0][0]

    return age


def face_detector(detector = 'mtcnn' ):
    """
    Initializes and returns the specified face detector.

    Parameters:
    detector (str): The type of face detector to use. Supported values are 'mtcnn' and 'haar'.

    detector = face_detector('mtcnn')  # For MTCNN detector
    detector = face_detector('haar')   # For Haar Cascade detector
    """
    if detector == 'mtcnn':
        from mtcnn import MTCNN
        return MTCNN()
    elif detector == 'haar':
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # elif detector == 'yolo':
    #     pass
    else:
        raise ValueError(f"Unknown detector: {detector}. Please choose supported detector.")


# Initialise face classifier
detector = 'mtcnn'  # Choose from available detectors, default = 'mtcnn'
face_det = face_detector(detector)

# Initiate the NN model
model_path = 'FastViT.h5'
model = preload_model(model_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")

frame_skip = 3  # Process every n-th frame
frame_count = 0
face_boxes = None

while True:
    ret, frame = camera.read()

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    if detector == 'haar':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_boxes = [{'box': (x, y, w, h)} for (x, y, w, h) in faces]
    elif detector == 'mtcnn':
        faces = face_det.detect_faces(frame)
        face_boxes = faces
    else:
        raise ValueError(f"Unknown detector: {detector}. Please choose supported detector.")

    for face in face_boxes:
        x, y, w, h = face['box']
        # Extend bbox to capture the whole face
        x_ext = max(0, x - int(0.125 * w))
        y_ext = max(0, y - int(0.125 * h))
        w_ext = min(frame.shape[1], x + w + int(0.25 * w)) - x_ext
        h_ext = min(frame.shape[0], y + h + int(0.25 * h)) - y_ext

        face_roi = frame[y_ext:y_ext + h_ext, x_ext:x_ext + w_ext]  # Face region extraction
        age = predict_age(face_roi, model)  # Model prediction

        # Bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # Bbox whole face
        cv2.rectangle(frame, (x_ext, y_ext), (x_ext + w_ext, y_ext + h_ext), (0, 255, 0), 3)

        # Display age and gender
        label = f"{int(age)}"
        cv2.putText(frame, label, (x_ext, y_ext - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    cv2.imshow('Webcam Age Gender Prediction', frame)

    # Exit the webcam window when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

