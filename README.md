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
- `pages/signin.py` — Firebase authentication page (sign in / sign up)
- `skin_cancer_model.h5` — trained classification model
- `.env` — local environment variables (not committed)
- `env.example` — template for environment variables

## Firebase Setup

This project uses Firebase Authentication for the sign-in/sign-up flow.

### 1. Create a Firebase project

1. Go to [https://console.firebase.google.com](https://console.firebase.google.com)
2. Click **Add project** and follow the steps
3. In the left sidebar go to **Build → Authentication**
4. Click **Get started** and enable the **Email/Password** sign-in provider

### 2. Get your Firebase config

1. In the Firebase console go to **Project settings** (gear icon)
2. Under **Your apps** click the **Web** icon (`</>`) to register a web app
3. Copy the config values shown

### 3. Set up your `.env` file

Copy `env.example` to `.env` and fill in your Firebase values:

```bash
cp env.example .env
```

```env
FIREBASE_API_KEY=your_api_key_here
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
FIREBASE_MESSAGING_SENDER_ID=your_sender_id
FIREBASE_APP_ID=your_app_id
FIREBASE_MEASUREMENT_ID=your_measurement_id
FIREBASE_DATABASE_URL=
```

> **Note:** `.env` is listed in `.gitignore` and will never be committed. Never share or commit this file.

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
