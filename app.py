import streamlit as st
import numpy as np
from PIL import Image
from inference import predict
from preprocessing import preprocess

classes = ["AKIEC","BCC","BKL","DF","MEL","NV","VASC"]

st.title("SmartDerma – Skin Cancer Detection")

file = st.file_uploader("Upload dermoscopic image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    img_np = np.array(img)
    processed = preprocess(img_np)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, width=300)

    with col2:
        st.image(processed, width=300)

    pred, overlay = predict(processed)

    st.success(f"Prediction: {classes[pred]}")
    st.image(overlay, width=300)
