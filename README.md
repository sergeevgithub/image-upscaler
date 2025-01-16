# ↕️ Image Upscaler App

Web application for upscaling images using a Super Resolution Generative Adversarial Network (SRGAN). 

The app is built with Streamlit.

The application currently supports image upscaling for input sizes of up to 256x256 pixels. This limitation is due to computational constraints in the current implementation.

## System Design

[System Design Document](https://docs.google.com/document/d/1-Br9nFJZ-XVLywGMs1I3CzDxwQuCXnSiAHS7YWpm1tE/edit?tab=t.0)

<img width="1000" src="https://github.com/user-attachments/assets/73248bd8-e856-4cc8-92dd-3d7a557534c9" />

## Use the app online

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlsd-image-upscaler.streamlit.app/)

## Local Installation and Setup
If you want to improve the app or use local GPU for faster processing,
follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```
   git clone https://github.com/sergeevgithub/image-upscaler.git
   cd image-upscaler
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Acknowledgments

* The GAN model used in this project is based on the Photo-Realistic Single Image Super-Resolution Generative Adversarial
Network ([SRGAN](https://arxiv.org/pdf/1609.04802v5)) architecture. 
