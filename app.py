import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

from model.gen import Generator


# Load the GAN model
@st.cache_resource
def load_model():
    # Adjust to gpu after Google Cloud deployment
    model = Generator()
    checkpoint = torch.load('model/weights/gen_weights.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Load the model
model = load_model()

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# Image transformation
def upscale_image(image, model):
    # Transform the image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        upscaled_tensor = model(input_tensor).squeeze()

    # Convert the output tensor back to an image
    upscaled_image = transforms.ToPILImage()(denormalize(upscaled_tensor))
    return upscaled_image


# Streamlit interface
st.title(" ↕️ Image Upscaling")

st.markdown("Upload an image to upscale it using a pretrained GAN model.")
st.markdown("Currently the app supports an input of sizes up to 256x256.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_container_width=True)
    st.error(f"🚫 The uploaded image size ({width}x{height}) exceeds the limit of 256x256 pixels. Please upload a smaller image.")
    # Check the dimensions of the uploaded image
    width, height = input_image.size
    if width > 256 or height > 256:
        st.error(f"🚫 The uploaded image size ({width}x{height}) exceeds the limit of 256x256 pixels. Please upload a smaller image.")
    else:
        # Process the image
        with st.spinner("Upscaling..."):
            upscaled_image = upscale_image(input_image, model)
    
        # Display the upscaled image
        st.image(upscaled_image, caption="Upscaled Image", use_container_width=True)
    
        # Option to download the upscaled image
        st.markdown("### Download the result")
        upscaled_image.save("upscaled_image.png")
        with open("upscaled_image.png", "rb") as file:
            btn = st.download_button(
                label=" ⏬ Download Image",
                data=file,
                file_name="upscaled_image.png",
                mime="image/png"
            )
    
        st.markdown("Rate the result!")
        sentiment_mapping = ["one", "two", "three", "four", "five"]
        selected = st.feedback("stars")
        if selected is not None:
            st.markdown("Thank you for the feedback!")
