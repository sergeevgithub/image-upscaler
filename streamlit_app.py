import streamlit as st
from PIL import Image

# Streamlit interface
st.title("Image Upscaling with GAN")

st.markdown("Upload an image to upscale it using a pretrained GAN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    # Process the image
    with st.spinner("Upscaling..."):
        upscaled_image = input_image

    # Display the upscaled image
    st.image(upscaled_image, caption="Upscaled Image", use_container_width=True)

    # Option to download the upscaled image
    st.markdown("### Download the Upscaled Image")
    upscaled_image.save("upscaled_image.png")
    with open("upscaled_image.png", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="upscaled_image.png",
            mime="image/png"
        )
