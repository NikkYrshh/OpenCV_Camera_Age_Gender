import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def preload_model(path):
    """ Load and Initialise the model """
    model = load_model(path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])

    return model


def pre_process_image(img):
    """ Pre-process the image for neural network """
    img = img.astype(np.float32)
    pp_image = cv2.resize(img, (128, 128))  # resize image
    pp_image = pp_image / 255.  # normalise image
    pp_image = np.expand_dims(pp_image, axis=0)  # add batch dimension

    return pp_image


def predict_age_gender(img, model):
    """Predict """
    image_pp = pre_process_image(img)
    preds = model.predict(image_pp)
    age = preds[0][0][0]
    gender_prob = preds[1][0][0]
    gender = 'Male' if gender_prob < 0.5 else 'Female'

    return age, gender


# # Initialise face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initiate the NN model
model_path = 'age_gender_A.h5'
model = preload_model(model_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]  # Face region extraction
        age, gender = predict_age_gender(face_roi, model)  # Model prediction

        # Bounding Box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display age and gender
        label = f"{gender}, {int(age)}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Webcam Age Gender Prediction', frame)

    # Exit the webcam window when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

