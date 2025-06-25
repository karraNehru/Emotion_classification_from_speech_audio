
# 🎧 Speech Emotion Recognition using EfficientNet-B3

This project performs emotion classification from speech audio by converting them into **log-mel spectrograms** and training an **EfficientNet-B3** deep learning model.

---

## 📁 Project Structure

```
├── mel_spectrograms/         # Folder of log-mel spectrogram images for training
├── predict.py                # Script to predict emotion from audio file
├── train_model.ipynb         # Jupyter notebook for training the model
├── emotion_b3_model.pth      # Trained model file (EfficientNet-B3)
└── README.md                 # Project overview and usage guide
```

---

## 🔍 Problem Statement

Classify emotional states from speech audio into one of the following classes:

- angry
- calm
- disgust
- fearful
- happy
- neutral
- sad
- surprised

### 🎯 Objective:
- Overall accuracy > **80%**
- F1-score > **80%**
- Per-class accuracy > **75%**

---

## 🧠 Model Details

- **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
- **Input size**: 300×300 RGB log-mel spectrograms
- **Loss**: CrossEntropy with label smoothing & class weights
- **Optimizer**: Adam (`lr=5e-5`)
- **Scheduler**: ReduceLROnPlateau
- **Augmentations**: Resize, Flip, Affine, Jitter

---

## 📦 Dependencies

Install requirements:
```bash
pip install torch torchvision librosa matplotlib scikit-learn seaborn
```

---

## 🏋️‍♂️ Training

Run the training in Jupyter:
```python
# At the end of training
torch.save(model.state_dict(), "emotion_b3_model.pth")
```

---

## 🔊 Predicting Emotion from Audio

Run the following command in terminal:

```bash
python predict.py sample.wav emotion_b3_model.pth
```

Output:
```
🎤 Predicted Emotion: happy
```

---

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (per class)
- Confusion Matrix plotted using `seaborn.heatmap`

---

## 📌 Notes

- Spectrograms are generated using **log-mel** scale with `librosa`.
- Normalize using `[0.5, 0.5, 0.5]` in transforms.
- Model expects RGB image input of size **300×300**.

---

## 🧠 Future Work

- Add streaming prediction (real-time mic input)
- Improve using attention or transformer blocks
- Build a Streamlit GUI interface

---

## 👤 Author
- Arjun Chawan
