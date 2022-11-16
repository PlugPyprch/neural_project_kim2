import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

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

uploaded_file = st.file_uploader("Choose a image file", type=['jpg', 'png', 'ipeg'])

class_names = ['MildDemented',
               'ModerateDemented',
               'NonDemented',
               'VeryMildDemented']

def load_image(image_file):
    img = Image.open(image_file)
    resized = img.resize((224,224))
    return resized


if uploaded_file is not None:
    resized = load_image(uploaded_file)
    st.image(resized, channels="RGB")

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(np.expand_dims(resized, axis=0))
        st.write('Predicted Label for the image is : ')
        st.title("{}".format(class_names[prediction.argmax()]))
