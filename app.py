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

    # Assuming your model outputs probabilities for 2 classes (tumor, no tumor)
    if predictions.shape[1] == 1:  # Binary classification
        predicted_class = (predictions[0][0] > 0.5).astype(int)  # Thresholding for binary classification
        class_labels = ['No Tumor', 'Tumor']
    else:
        # Handle multi-class case if applicable
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4']  # Adjust based on your model

    string = "Detected class: " + class_labels[predicted_class]

    if predicted_class == 1:  # Assuming 1 corresponds to 'Tumor'
        st.balloons()
        st.sidebar.success(string)
        st.write("""
        # Tumor Detected! 🧠
        """)
    else:
        st.sidebar.warning(string)
        st.markdown("## No Tumor Detected")
