# 👄 Lip Reading AI

Upload a silent video and watch AI read the lips — extracting lip movements frame by frame using MediaPipe and predicting speech using Claude Vision AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?style=flat-square)
![Claude Vision](https://img.shields.io/badge/Claude-Vision_API-orange?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-blue?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What It Does

- 🎬 **Upload any video** (MP4, MOV, AVI, WEBM)
- 🔬 **Extracts frames** — samples up to 20 frames from the video
- 👄 **Detects lip landmarks** — MediaPipe tracks 40 lip points per frame
- ✂️ **Crops lip region** — precise bounding box around the mouth
- 🧠 **Claude Vision analyzes** — sends lip frame sequence to AI for speech prediction
- 📊 **Full analysis report** — confidence score, phonemes, alternative predictions
- 🖼️ **Visual output** — shows extracted lip crops + annotated landmark frames

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| MediaPipe Face Mesh | 468 facial landmark detection |
| OpenCV | Video processing + frame extraction |
| Claude Vision API | Multimodal lip movement analysis |
| Pillow | Image resizing + encoding |
| Streamlit | Web app UI |

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
8 lip frame crops → base64 encoded
     ↓
Claude Vision API → analyze lip movement sequence
     ↓
Predicted speech + confidence + phonemes displayed
```

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/Navld14/lip-reading-ai.git
cd lip-reading-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Get your free API key at [console.anthropic.com](https://console.anthropic.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🎬 Tips for Best Results

- Face should be clearly visible and well-lit
- Speaker facing the camera directly (front-facing)
- Short clips of 2–10 seconds work best
- Avoid fast head movements or motion blur
- Higher resolution videos give better landmark detection

---

## 📁 Project Structure

```
lip-reading-ai/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## 🙋 Author

**Naveen Ladha**
[GitHub](https://github.com/Navld14)

---

## 📄 License

MIT License — free to use and modify.
