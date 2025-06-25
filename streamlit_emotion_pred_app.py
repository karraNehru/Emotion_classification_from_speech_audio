import streamlit as st
import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io

# Define emotion classes
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Load model
@st.cache_resource

def load_model(model_path, device):
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Convert audio to image
@st.cache_data

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    librosa.display.specshow(mel_spec_db, sr=sr, cmap='magma')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image

# Streamlit App
st.set_page_config(page_title="Speech Emotion Classifier", layout="centered")
st.title(" speech Emotion Recognition")
st.markdown("Upload a `.wav` file and let the model predict the emotion in the voice.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file:
    with st.spinner('Analyzing emotion...'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model("efficientnet_b2_emotion.pth", device)

        image = preprocess_audio(uploaded_file)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            confidence = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(confidence)
            predicted_emotion = EMOTION_LABELS[pred_idx]
            confidence_score = confidence[pred_idx] * 100

        st.success(f"Predicted Emotion: **{predicted_emotion}**")
        st.markdown(f"Confidence Score: **{confidence_score:.2f}%**")

        st.image(image, caption="Mel-Spectrogram", use_container_width=True)

        st.markdown("---")
        st.subheader("Confidence per Emotion")
        for idx, label in enumerate(EMOTION_LABELS):
            st.progress(int(confidence[idx]*100), text=f"{label}: {confidence[idx]*100:.2f}%")
