import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import SimpleCNN
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)


# Load Model
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# CIFAR-10 class labels
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])


# UI Header

st.markdown(
    """
    <h1 style="text-align:center; color:#4CAF50;">
        CIFAR-10 Image Classification System
    </h1>
    <p style="text-align:center; font-size:18px;">
        Upload an image and let the AI classify it into one of the 10 CIFAR categories.
    </p>
    """,
    unsafe_allow_html=True
)


# Image Upload

file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "png"])


# Predict Function
def predict(image):
    img = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
    return probabilities



# Main Logic
if file:
    image = Image.open(file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get model prediction
    probabilities = predict(image)
    pred_idx = np.argmax(probabilities)
    pred_label = class_names[pred_idx]
    pred_conf = probabilities[pred_idx] * 100

    
    #  Result Card
    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f0f8f5; text-align:center; border:2px solid #4CAF50;">
            <h2 style="color:#2E7D32;">Prediction: {pred_label}</h2>
            <h3 style="color:#388E3C;">Confidence: {pred_conf:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Confidence Bar Chart
    st.subheader(" Confidence Score Chart")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, probabilities)
    ax.set_ylabel("Confidence")
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    st.pyplot(fig)

else:
    st.info("Please upload an image above to begin classification.")
