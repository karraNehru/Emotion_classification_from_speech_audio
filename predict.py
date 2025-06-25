
#script file for predicting on input data

#importing packages
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import sys

#List of emotion classes
EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

#Convert audio to Mel-Spectrogram image
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



def load_model(model_path, device):
    model = models.efficientnet_b2(weights=None)  # Use torchvision's version
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)  # 8 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


#Predict Emotion
def predict_emotion(audio_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = preprocess_audio(audio_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    model = load_model(model_path, device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    predicted_emotion = EMOTION_LABELS[pred_idx]
    print(f"🎤 Predicted Emotion: {predicted_emotion}")

#Entry point
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <audio_file.wav> <model_file.pth>")
    else:
        audio_path = sys.argv[1]
        model_path = sys.argv[2]
        predict_emotion(audio_path, model_path)
