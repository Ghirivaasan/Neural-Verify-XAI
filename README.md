# 🕵️ Multimodal Fake News Credibility Scoring

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

A state-of-the-art Deep Learning system that effectively combats misinformation by analyzing the specific alignment—or lack thereof—between a news article's text and its accompanying image.

## 📖 The Problem
Modern "Fake News" often utilizes a technique called "Out-of-Context media." Publishers take a terrifying, provocative image (e.g., a massive tsunami from 2011) and attach it to a completely unrelated headline (e.g., "Tsunami hits Florida Coast Today!"). Since standard NLP models only read text, they fail to recognize this discrepancy. 

## 🧠 The Architecture (Dual-Stream Model)
This project implements a sophisticated **Dual-Stream** Deep Learning model alongside advanced forensics:
1. **Feature Extraction (CLIP)**: We utilize OpenAI’s CLIP (`clip-vit-base-patch32`) to project both Text and Images into a mathematically shared embedding space.
2. **Consistency Module**: 
   - Calculates **Cosine Similarity** to quantify high-level vector alignment.
   - Computes **Multi-Head Cross-Attention** where Text vectors query visual patches, asking: "Does this specific word exist anywhere in this image?"
3. **Fusion & Classification**: The Multi-Layer Perceptron (MLP) concatenates the text vector, image vector, and the consistency scores, passing them through `Dense -> ReLU -> Dropout` layers. A final `Sigmoid` activation crushes the output into an intuitive `0 to 1` credibility score.
4. **Zero-Shot NLP Sensationalism**: Re-uses the underlying textual embeddings to mathematically measure the distance between the uploaded text and anchor prompts indicating "Sensational/Clickbait" writing.
5. **Digital Forensics (ELA)**: Includes an **Error Level Analysis** parser to dynamically compute non-uniform JPEG compression zones, highlighting artificially photoshopped overlays.

## 🚀 Key Features
- **Zero-Error PyTorch Implementation**: Well-structured `Dataset` and `Module` classes.
- **Explainable AI (XAI)**: Generates highly visual heatmaps in the frontend showcasing the semantic-visual conflicts detected by Cross-Attention layers.
- **Fakeddit Dataset Ingestion**: Custom scripts (`fakeddit_parser.py`) for processing multi-million row TSV datasets, safely downloading active image URLs seamlessly.
- **Glassmorphism Premium UI**: Includes a fully-functional, stunning `Streamlit` app featuring multi-tab views, responsive glowing gradients, and CSS injections built to mimic a high-end React dashboard.

---

## 💻 Installation and Setup

### 1. Requirements
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 2. Project Structure
```text
Multimodal-FakeNews-Detector/
├── src/
│   ├── dataset.py        # Custom PyTorch Dataset handling CLIP preprocessing
│   ├── model.py          # Core Dual-Stream Architecture w/ Cross-Attention
│   ├── train.py          # Training Loop with Mixed Precision
│   └── evaluate.py       # Evaluation and Diagnostic plotting
├── app.py                # Visually stunning Streamlit UI for inference
├── requirements.txt      # Project dependencies
└── README.md             # This document
```

### 3. Training the Model
To initiate a training run:
```bash
# By default, this runs a smoke test using randomly generated mock data to ensure no shape mismatch bugs exist.
python src/train.py
```
*Note: In production, modify the dataset instantiation inside `train.py` to point to a valid Pandas DataFrame (e.g., Fakeddit dataset).*

### 4. Running the Interactive Resume App
We provide an interactive demo application for testing your real-time text and user-uploaded images.
```bash
streamlit run app.py
```
This application will automatically load your trained weights from `checkpoints/best_model.pth`. If no trained model is found, it will run un-initialized features for demonstration purposes.

---

## 📊 Evaluation & Scores

| Score Range | Interpretation | Action Taken |
| ----------- | -------------- | ------------ |
| `0.0 - 0.3` | **Low Credibility** | Flagged as Fake News / Mismatched Content |
| `0.3 - 0.7` | **Uncertain** | Marked for human fact-checking |
| `0.7 - 1.0` | **High Credibility** | Verified; Text and Image are mathematically consistent |

Run the standard evaluation tools to generate your precision matrices:
```bash
python src/evaluate.py
```

*Built with ❤️ utilizing PyTorch & HuggingFace Ecosystems.*
