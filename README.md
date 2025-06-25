
# Emotion Classification from Speech audio

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

![Image](https://github.com/user-attachments/assets/84c5fe73-d1d8-43b7-93dd-925effefde1f)


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

![Image](https://github.com/user-attachments/assets/748454eb-8938-4c7f-8834-5798d5d7e51a)


## Model Architecture

- **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
- **Input size**: 300×300 RGB log-mel spectrograms
- **Loss**: CrossEntropy with label smoothing & class weights
- **Optimizer**: Adam (`lr=5e-5`)
- **Scheduler**: ReduceLROnPlateau
- **Augmentations**: Resize, Flip, Affine, Jitter

---



## Evaluation Metrics

| Metric          | Target       | Achieved   |
|-----------------|--------------|------------|
| Overall Accuracy| >80%         | ✅81%         |
| F1 Score        | >80%         |  ✅81%       |
| Per-class Acc   | >75%         | ✅Met for all classes except sad(74%) |

Includes detailed:
- Classification report
 ```bash
--- Classification Report ---
              precision    recall  f1-score   support

       angry       0.91      0.89      0.90        76
        calm       0.82      0.84      0.83        67
     disgust       0.72      0.84      0.78        31
     fearful       0.82      0.77      0.79        78
       happy       0.85      0.77      0.81        75
     neutral       0.74      0.78      0.76        41
         sad       0.77      0.74      0.75        84
   surprised       0.81      0.97      0.88        39

    accuracy                           0.81       491
   macro avg       0.81      0.83      0.81       491
weighted avg       0.82      0.81      0.81       491
```
**Confusion matrix:**
![Image](https://github.com/user-attachments/assets/9c47370b-bec8-4203-871d-9fa121f6e70d)


## How to Run
## 📦 Dependencies

Install requirements: from requirements.txt

### 🤖 Train the Model
```bash
# Inside Jupyter Notebook
Run the cells in `emotion_classification_2.ipynb`
```
Run the training in Jupyter:
```bash
# At the end of training
torch.save(model.state_dict(), "emotion_b3_model.pth")
```

### Predict Emotion on New Audio
```bash
# In Terminal or PowerShell
python predict.py <path_to_audio.wav> <model_path.pth>

# Example:
python predict.py 03-02-01-01-01-01-01.wav emotion_b3_model.pth
```
---

## 🌐 Streamlit Web App

This project includes an interactive Streamlit app that:
- Accepts `.wav` audio uploads
- Shows the predicted emotion

**App:**
[Real-time-emotion-predictor-app](https://real-time-emotion-predictor.streamlit.app/)

---

##  Project Structure

```
emotion_classification_project/
│
├── emotion_classification_2.ipynb   # Training notebook
├── predict.py                       # Inference script
├── emotion_b3_model.pth             # Trained model
├── requirements.txt                 # Dependencies
├── README.md                        # Project overview (this file)
├── app.py                           # Streamlit web app
```

---

## 📬 Contact

For any questions or collaboration, feel free to reach out.

---
