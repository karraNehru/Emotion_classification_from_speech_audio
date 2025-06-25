
# Emotion Classification on Speech Data

This project implements a robust end-to-end deep learning pipeline for emotion classification using speech/audio data. It leverages Mel-spectrograms and EfficientNet-B2 to accurately recognize human emotions conveyed in voice recordings.


## Problem statement
> Design and implement a robust pipeline to perform emotion classification on speech data.

The goal is to classify audio samples into one of several emotional categories (e.g., happy, sad, angry, etc.) using machine learning and audio signal processing techniques.

## Objective
- Convert raw audio files to Mel-spectrograms.
- Build a high-performance deep learning model using EfficientNet.
- Achieve >80% F1-score and per-class accuracy >75% on the validation set.
- Deliver a complete solution including inference script and web interface.

## Data set
From [This Data set](https://zenodo.org/records/1188976#.XCx-tc9KhQI)

The following files are used for the  dataset:
1. Audio_Speech_Actors_01-24
2. Audio_Song_Actors_01-24.

**Description:**

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The dataset contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in modality format: Audio-only (16bit, 48kHz .wav),  Note: there are no song files for Actor_18.





## Pre-processing steps

1. Emotion Mapping
```bash
filename = '03-01-01-01-01-01-01.wav'
             ^  ^  ^  ^  ^  ^  ^
             |  |  |  |  |  |  └─ actor ID
             |  |  |  |  |  └──── repetition
             |  |  |  |  └────── intensity
             |  |  |  └───────── statement
             |  |  └──────────── vocal channel
             |  └─────────────── emotion (01–08)
             └────────────────── modality

```
The third value (from left) is the emotion ID, which is mapped as:

```bash
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
```
2. Creating the DataFrame

- Traverse both folders:

   Audio_Speech_Actors_1-24

   Audio_Song_Actors_1-24

- Extract and record:

  file_path, emotion label, actor ID, and gender

- Combine into a single pandas DataFrame and saved to file ( )

3. Convert Audio to Mel-Spectrograms
- Each audio file is loaded using librosa
- Generated Mel-spectrogram




## Model Architecture

- **Backbone**: [EfficientNet-B2](https://arxiv.org/abs/1905.11946)
- **Modifications**: Replaced final FC layer with 8-class classifier
- **Loss Function**: Cross-entropy with optional label smoothing
- **Optimizer**: Adam
- **Training Framework**: PyTorch


## Evaluation Metrics

| Metric          | Target       | Achieved   |
|-----------------|--------------|------------|
| Overall Accuracy| >80%         | 76%         |
| F1 Score        | >80%         |  76%       |
| Per-class Acc   | >75%         | Met for all classes exept sad and  happy |

Includes detailed:
- Classification report
 ```bash
 Classification Report:

              precision    recall  f1-score   support

       angry       0.87      0.87      0.87        69
        calm       0.76      0.89      0.82        61
     disgust       0.72      0.83      0.77        35
     fearful       0.67      0.87      0.76        79
       happy       0.85      0.71      0.77        95
     neutral       0.67      0.79      0.72        43
         sad       0.77      0.47      0.59        76
   surprised       0.84      0.79      0.81        33

    accuracy                           0.76       491
   macro avg       0.77      0.78      0.76       491
weighted avg       0.77      0.76      0.76       491
```



## How to Run

### Train the Model
```bash
# Inside Jupyter Notebook
Run the cells in `train_emotion_classifier.ipynb`
```

### Predict Emotion on New Audio
```bash
# In Terminal or PowerShell
python predict.py <path_to_audio.wav> <model_path.pth>

# Example:
python predict.py 03-02-01-01-01-01-01.wav efficientnet_b2_emotion.pth
```
---

## Streamlit Web App

This project includes an interactive Streamlit app that:
- Accepts `.wav` audio uploads
- Shows the predicted emotion

(To be included in `app.py`)

---

##  Project Structure

```
emotion_classification_project/
│
├── train_emotion_classifier.ipynb   # Training notebook
├── predict.py                       # Inference script
├── efficientnet_b2_emotion.pth      # Trained model
├── requirements.txt                 # Dependencies
├── README.md                        # Project overview (this file)
├── app.py                           # Streamlit web app
```

---

## 📬 Contact

For any questions or collaboration, feel free to reach out.

---
