# Multimodal Emotion Recognition on MELD Dataset 

**An End-to-End Deep Learning System Fusing Speech and Text for Emotion Classification.**
*Includes an Interactive Web UI for real-time testing.*

This repository contains the implementation of the final project for the **Machine Learning** course. The project focuses on **Multimodal Emotion Recognition** using the **MELD** dataset. By leveraging **Late Fusion** of Audio and Text modalities, the model achieves superior performance compared to unimodal approaches, particularly in detecting complex emotions like sarcasm.

---

##  Project Overview
Human communication is multimodal. While text conveys semantics, audio carries prosody and tone. This project implements a **Dual-Branch Neural Network**:
1.  **Text Branch:** Uses **RoBERTa** (Transformer) to extract contextual semantic features.
2.  **Audio Branch:** Processes raw audio/features (e.g., using **Wav2Vec2** or spectral features) to capture acoustic characteristics.
3.  **Fusion:** Concatenates embeddings from both branches for final classification.

**Dataset:** [MELD (Multimodal EmotionLines Dataset)](https://github.com/declare-lab/MELD) containing dialogue utterances from the TV series *Friends*.

---

##  Key Features

*   ** Multimodal Fusion:** Combines Text and Audio inputs to predict 7 emotion classes (Anger, Disgust, Sadness, Joy, Neutral, Surprise, Fear).
*   ** Advanced Architectures:**
    *   **NLP:** Fine-tuning of **RoBERTa-base** for robust text embedding.
    *   **Speech:** Deep feature extraction aligned with text timestamps.
*   ** Ablation Study:** Comprehensive analysis comparing "Audio-Only", "Text-Only", and "Combined" models (Detailed in `MLP2-report.pdf`).
*   ** Interactive UI (Gradio):** A web-based interface allowing users to record voice or input text to test the model in real-time.

---

##  Project Structure

*   `p2 (1).ipynb`: The complete pipeline including:
    *   Data Preprocessing (Tokenization, Audio resampling).
    *   Custom Dataset & Dataloader implementation.
    *   Model Architecture Definition (RoBERTa + Audio Encoder).
    *   Training Loop with Validation.
    *   **Gradio UI Implementation**.
*   `MLP2-report.pdf`: Technical report covering architecture, hyperparameters, ablation studies, and error analysis.
*   `Project2.pdf`: Original project problem statement.

---

##  Methodology & Architecture

The model uses a **Late Fusion** strategy:

1.  **Input:** An utterance consisting of an Audio clip ($A$) and its Transcript ($T$).
2.  **Feature Extraction:**
    *   $E_T = \text{RoBERTa}(T)$
    *   $E_A = \text{AudioEncoder}(A)$
3.  **Fusion:**
    *   $V_{fused} = \text{Concat}(E_T, E_A)$
4.  **Classification:**
    *   $y_{pred} = \text{Softmax}(\text{MLP}(V_{fused}))$
---

##  Installation & Usage

### 1. Prerequisites
Install the required libraries, including `gradio` for the UI:
```bash
pip install torch torchvision transformers librosa numpy pandas scikit-learn tqdm soundfile gradio
```
### 2. Running the Training
Open the notebook and run the cells sequentially to train the model:
```bash
jupyter notebook "p2 (1).ipynb"
```
### 3. Running the Interactive Demo (UI)
The last section of the notebook contains the **Gradio** code. Running that cell will launch a local web server:
<img width="1269" height="572" alt="image" src="https://github.com/user-attachments/assets/927cb81f-6fee-43db-b087-0912478505a1" />
