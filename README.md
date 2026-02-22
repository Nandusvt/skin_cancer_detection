# Skin Cancer Detection AI (Streamlit)

A Streamlit web app for skin lesion classification using a trained TensorFlow/Keras model (`skin_cancer_model.h5`).

## Features

- Upload lesion images (`.jpg`, `.jpeg`, `.png`)
- Predicts 1 of 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- Displays confidence score and class probabilities
- Shows disease details, risk factors, and treatment notes
- Downloadable analysis report
- Includes medical disclaimer for safe usage

## Project Structure

- `app.py` — Streamlit frontend + inference pipeline
- `skin_cancer_model.h5` — trained classification model

## Quick Start (Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Dependencies

All dependencies are listed in `requirements.txt`.

## Notes

- This app is for educational/screening use only.
- It is **not** a medical diagnosis tool.
- Always consult a qualified dermatologist for clinical decisions.
