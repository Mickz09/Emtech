# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleCNN 
# 1) helper: load model
@st.cache_resource
def load_model(weights_path="model_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

# 2) prediction helper
def predict_image(model, device, pil_image):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img_t = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        pred = out.argmax(dim=1).item()
    return pred

st.title("CIFAR-10 Demo (Streamlit + PyTorch)")
st.write("Upload an image to classify (CIFAR-10 classes)")

uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])
model, device = load_model("model_weights.pth")

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    pred = predict_image(model, device, image)
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    st.success(f"Predicted: {classes[pred]}")
