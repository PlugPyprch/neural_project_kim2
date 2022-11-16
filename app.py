import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

st.markdown("""
<style>
p {
    font-size:1.2rem !important;
    margin: 0 0 0 0;
}
h1 {
    padding: 0 0 0 0;
}

</style>
""", unsafe_allow_html=True)

st.title('Stage of Alzheimer You Are ? 🧠')

model = load_model("./model_al.h5")

def load_and_prep_image(filename, image_shape=224, scale=True):
  img = image.load_img(filename, target_size=(image_shape, image_shape))
  img = np.asarray(img)
  return img/255.

uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'png', 'ipeg'])

class_names = ['MildDemented',
               'ModerateDemented',
               'NonDemented',
               'VeryMildDemented']


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = cv2.resize(opencv_image,(224,224))

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(np.expand_dims(resized, axis=0))
        st.write('Predicted Label for the image is : v')
        st.title("{}".format(class_names[prediction.argmax()]))
