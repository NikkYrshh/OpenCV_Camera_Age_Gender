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
    gender = 'Male' if gender_prob <0.5 else 'Female'

    return age, gender

img_path = 'test_image.jpg'

img = plt.imread(img_path)

# plt.imshow(img)
# plt.show()

model_path = 'age_gender_A.h5'
model = preload_model(model_path)
age, gender = predict_age_gender(img, model)
print(age, " ", gender)

