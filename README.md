# 👄 Lip Reading AI

Upload a silent video and watch AI read the lips — extracting lip movements frame by frame using MediaPipe and predicting speech using Google Gemini Vision AI.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Google-Gemini_Vision-4285F4?style=flat-square&logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-blue?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

🌐 **Live Demo:** [Try it here](https://lip-reading-ai-h578.onrender.com/)

> ⚠️ First load may take 30-50 seconds as the server wakes from idle (Render free tier).

---

## 🎯 What It Does

- 🎬 **Upload any video** (MP4, MOV, AVI, WEBM)
- 🔬 **Extracts frames** — samples up to 20 frames from the video
- 👄 **Detects lip landmarks** — MediaPipe tracks 40 lip points per frame
- ✂️ **Crops lip region** — precise bounding box around the mouth
- 🧠 **Gemini Vision analyzes** — sends lip frame sequence to AI for speech prediction
- 📊 **Full analysis report** — confidence score, phonemes, alternative predictions
- 🖼️ **Visual output** — shows extracted lip crops + annotated landmark frames

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| MediaPipe | 468 facial landmark detection |
| OpenCV | Video processing + frame extraction |
| Google Gemini Vision API | Multimodal lip movement analysis (free tier) |
| Pillow | Image resizing + encoding |
| Streamlit | Web app UI |
| Docker | Containerized deployment on Render |

---

## 💡 How It Works

```
Video Upload
     ↓
OpenCV → extract frames (up to 20)
     ↓
MediaPipe Face Mesh → detect 468 facial landmarks
     ↓
40 lip landmarks → bounding box → crop lip region
     ↓
8 lip frame crops → bytes encoded
     ↓
Gemini Vision API → analyze lip movement sequence
     ↓
Predicted speech + confidence + phonemes displayed
```

---

## 🚀 Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Navld14/Lip-Reading-AI.git
cd Lip-Reading-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Gemini API key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **Get API Key → Create API key**

### 4. Set your API key
```bash
export GEMINI_API_KEY="AIzaSy..."
```

### 5. Run the app
```bash
streamlit run app.py
```

### Or run with Docker
```bash
docker build -t lip-reading-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key lip-reading-ai
```

---

## 🎬 Tips for Best Results

- Face should be clearly visible and well-lit
- Speaker facing the camera directly (front-facing)
- Short clips of 2–10 seconds work best
- Avoid fast head movements or motion blur

---

## 📁 Project Structure

```
Lip-Reading-AI/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker config for deployment
└── README.md
```

---

## 🙋 Author

**Naveen Ladha**
[GitHub](https://github.com/Navld14)

---

## 📄 License

MIT License — free to use and modify.
