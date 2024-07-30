import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

with st.spinner("Loading model..."):
          EmotionModel = tf.keras.models.load_model('Model.keras')

st.toast('Model Ready!', icon='🎉')

def resize_and_add_dimension(image, target_size=(48, 48)):
    resized_image = cv2.resize(image, target_size)
    image_with_dimension = np.expand_dims(resized_image, axis=-1)
    return image_with_dimension

def preprocess(X):
          X = np.array([np.fromstring(image, np.uint8, sep=' ') for image in X])
          X = X/255.0
          X = X.reshape(-1, 48, 48, 1)
          return X

def preprocessnew(image):
    # Resize the image while preserving its aspect ratio
    resized_image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Add an extra dimension to match the desired shape (48, 48, 1)
    image_with_dimension = np.expand_dims(gray_image, axis=-1)
    
    # Normalize pixel values to [0, 1]
    image_with_dimension = image_with_dimension / 255.0

    image_with_dimension = image_with_dimension.reshape(-1,48,48,1)

    print(image_with_dimension.shape)
    
    return image_with_dimension

def check_if_have_face(image):
          face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
          if len(faces) == 0:
                    return None 
          for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
          return image

st.title("Facial Emotion Recognition, Use a photo to know what you're feeling!")
st.header("You'll get one of 7 results, make sure to vote if it was yours or not")
st.write("Just click upload below and check your emotion!")

image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if image is not None:
          with st.spinner("Checking if the image contains a face..."):
                    image = Image.open(image)
                    image = image.convert('RGB')
                    image = image.save('img.jpg')
                    image = cv2.cvtColor(cv2.imread('img.jpg'), cv2.COLOR_BGR2RGB)
                    face_det = check_if_have_face(image)
                    if face_det is None:
                        st.toast('This image contains no face!', icon='❌')
                    else: 
                              st.toast("This image contains face!", icon='🎉')
                              predValue = np.argmax(EmotionModel.predict(preprocessnew(image)))
                              st.image(face_det)
                              if predValue == 0:
                                    st.write("You are Angry! 😡")
                              elif predValue == 1:
                                      st.write("You are Disgusted! 🤮")
                              elif predValue == 2:
                                      st.write("You are Scared! 🫣")
                              elif predValue == 3:
                                      st.write("You are Happy 😃")
                              elif predValue == 4:
                                      st.write("You are Sad. 😭")
                              elif predValue == 5:
                                      st.write("You are Saurprised! 🫨")
                              elif predValue == 6:
                                      st.write("You are neutral. 😐")
                            

with st.sidebar:
          st.title('Project Information')
          st.header('Project Information:')
          st.subheader("This project was done using a model from scratch, after preprocessing the data as such [Using HaarCascades to check if a face exists in image; Augmenting the dataset to improve the size of the skewed classes via rotation and blurring], where the model on trained on the new data size of 100k datapoints.")
          st.image(cv2.imread('Vision.png'))
          with st.spinner("Calculating accuracy..."):
                    data = pd.read_csv('fer2013.csv')
                    groups = [g for _, g in data.groupby('Usage')]
                    val = groups[1]
                    test = groups[0]

                    val = val.drop(labels=['Usage'], axis=1)
                    test = test.drop(labels=['Usage'], axis=1)

                    Y_val = val["emotion"]
                    Y_test = test["emotion"]
                    #'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'

                    X_val = val["pixels"]
                    X_test = test["pixels"]

                    X_val = preprocess(X_val)
                    X_test = preprocess(X_test)
                    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=7)
                    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=7)
                    testAcc=EmotionModel.evaluate(X_test, Y_test) #Acc:65%
                    ValAcc=EmotionModel.evaluate(X_val, Y_val) #Acc:63%

          st.toast('Model Tested!', icon='🎉')
          st.write(f'Test Loss: {round(testAcc[0],2)}')
          st.write(f'Test Accuracy: {(round(testAcc[1]*100,2))}%')
          st.write(f'Validation Loss: {round(ValAcc[0])}')
          st.write(f'Validation Accuracy: {round(ValAcc[1]*100,2)}%')
          st.write("This project was done by:" )
          st.write('Youssef Mohamed Bedair - 20100260')
          st.write('Omar El-Hamrawy - 20100357')
          st.write('Mohamed Yasser - 20100"')
