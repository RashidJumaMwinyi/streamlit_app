import os
import gdown
import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def predict_image_class(image_data, model, w=128, h=128):
    size = (w, h)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = img[:, :, :3]  # Slice off the alpha channel if it exists
    img = np.expand_dims(img, axis=0)  # for models expecting a batch
    prediction = model.predict(img)
    return prediction

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.keras')  # Ensure the model filename matches
    return model

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    initial_sidebar_state='auto'
)

with st.sidebar:
    st.title("Brain Tumor Detection Model")
    st.subheader("This model detects the presence of brain tumors in MRI images.")

st.write("""
         # Brain Tumor Detection Tool
         """
         )

img_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])

model_file = 'my_model.keras'
if not os.path.exists(model_file):
    with st.spinner('Model is being downloaded...'):
        gdown.download(f'https://drive.google.com/uc?id=1ehLrhvVLb0a7QdGUQk5VGuQfneUVkXQo', output=model_file, quiet=False)

with st.spinner('Model is being loaded...'):
    model = load_model()

if img_file is None:
    st.text("Please upload an MRI image file.")
else:
    image = Image.open(img_file)
    st.image(image, use_container_width=False)
    predictions = predict_image_class(image, model)

    # Assuming your model outputs probabilities for 4 classes
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Class labels

    string = "Detected class: " + class_labels[predicted_class]
    
     # Add a link to download the dataset
    st.markdown(
        "If you don't have an MRI image to test, you can download the dataset from [here](https://drive.google.com/drive/folders/1sBfPzjdDLOup-whgQliKOhFa-adjKyiX?usp=sharing)."
    )
    
    # Display the result with different colors based on the detected class
    if class_labels[predicted_class] == 'notumor':
        st.sidebar.info(string)  # Use info for 'notumor'
        st.write(f"## Detected Class: {class_labels[predicted_class]}")
        st.success("No tumor detected. The MRI appears normal.")
    else:
        st.sidebar.error(string)  # Use error for tumor cases
        st.write(f"## Detected Class: {class_labels[predicted_class]}")
        st.warning("Tumor detected! Please consult a medical professional for further evaluation.")
