import os
import keras
import numpy as np
import tensorflow as tf
from keras.models import load_model
import streamlit as st

st.header('Image Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = load_model('Flower_recog_model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array=tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])
    outcome = 'This image is ' + flower_names[np.argmax(result)]+' with similarity score '+str(np.max(result)*100)
    return outcome

st.title("Image Recognition App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display image and make prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("Classifying...")
    
    result = classify_images(uploaded_file)
    st.markdown(result)
else:
    st.write("Please upload an image.")