import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import os
import anthropic
import base64
import io
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lip Reading AI 👄",
    page_icon="👄",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ec4899, #f43f5e, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p { color: #888; font-size: 1rem; margin-top: -6px; }

.result-card {
    background: linear-gradient(135deg, #0f0014, #1a0025);
    border: 1px solid #ec489944;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 12px 0;
}
.result-text {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f9a8d4;
    line-height: 1.3;
    letter-spacing: 2px;
}
.result-label {
    font-size: 0.7rem;
    color: #9f6aad;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

.confidence-card {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

.frame-strip {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 8px 0;
}

.stat-box {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.stat-value { font-size: 1.8rem; font-weight: 700; color: #ec4899; }
.stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 10px;
}

.lip-frame-info {
    background: #0d0014;
    border: 1px solid #ec489933;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #f9a8d4;
    margin-top: 8px;
}

.analysis-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #1a1a1a;
    color: #ccc;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── MediaPipe face mesh ────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# Lip landmark indices in MediaPipe Face Mesh
LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


# ── Extract frames from video ──────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = 20) -> list:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // max_frames)
    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames, fps, total_frames


# ── Crop lip region from frame ─────────────────────────────────────────────────
def crop_lip_region(frame, padding: int = 20):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    lip_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LIP_LANDMARKS]

    xs = [p[0] for p in lip_points]
    ys = [p[1] for p in lip_points]

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)

    lip_crop = rgb[y1:y2, x1:x2]

    # Draw on original
    annotated = rgb.copy()
    for point in lip_points:
        cv2.circle(annotated, point, 1, (236, 72, 153), -1)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (236, 72, 153), 2)

    return lip_crop, annotated


# ── Encode image to base64 ─────────────────────────────────────────────────────
def img_to_b64(img_array) -> str:
    pil = Image.fromarray(img_array.astype(np.uint8))
    pil = pil.resize((128, 64))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=85)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


# ── Claude vision lip reading ──────────────────────────────────────────────────
def analyze_lips_with_claude(lip_frames: list) -> dict:
    client = anthropic.Anthropic()

    # Use up to 8 evenly spaced frames for analysis
    n = len(lip_frames)
    indices = [int(i * (n - 1) / 7) for i in range(min(8, n))]
    selected = [lip_frames[i] for i in indices]

    content = []
    for i, frame in enumerate(selected):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_to_b64(frame)
            }
        })

    content.append({
        "type": "text",
        "text": """You are an expert lip reading AI. I'm showing you a sequence of lip/mouth crop images extracted from a video in chronological order.

Analyze the lip movements across these frames and determine what the person is saying.

Look for:
- Mouth shape changes (open/closed, rounded, spread)
- Lip movement patterns characteristic of specific phonemes
- Sequence of mouth positions to identify words or phrases

Respond ONLY with valid JSON (no markdown):
{
  "predicted_text": "<your best prediction of what was said>",
  "confidence": "<High | Medium | Low>",
  "confidence_score": <integer 1-100>,
  "phonemes_detected": ["<phoneme1>", "<phoneme2>"],
  "lip_activity": "<Minimal | Low | Moderate | High | Very High>",
  "mouth_movements": "<brief description of what you observed in the lip movements>",
  "alternative_predictions": ["<alt prediction 1>", "<alt prediction 2>"],
  "notes": "<any observations about video quality, face visibility, or confidence factors>"
}

If lips are not clearly visible or movement is too subtle, set confidence to Low and predicted_text to your best guess anyway."""
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": content}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>👄 Lip Reading AI</h1>
    <p>Upload a video — AI reads the lips and predicts what was said</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown("<div class='section-label'>🎬 Upload Video</div>", unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "",
        type=["mp4", "mov", "avi", "webm"],
        label_visibility="collapsed"
    )

    st.info("""
    **Tips for best results:**
    - Face clearly visible and well-lit
    - Speaker facing the camera directly
    - Short clips (2–10 seconds) work best
    - Avoid fast movements or blur
    """)

    if uploaded_video:
        st.video(uploaded_video)

        if st.button("👄 Read Lips", type="primary", use_container_width=True):
            # Save video to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            try:
                # Step 1 — Extract frames
                with st.status("Processing video...", expanded=True) as status:
                    st.write("🎬 Extracting frames from video...")
                    result = extract_frames(tmp_path, max_frames=20)

                    if not result or len(result[0]) == 0:
                        st.error("Could not extract frames from video.")
                        st.stop()

                    frames, fps, total_frames = result
                    st.write(f"✅ Extracted {len(frames)} frames (video: {total_frames} total @ {fps:.1f} fps)")

                    # Step 2 — Detect and crop lips
                    st.write("👄 Detecting lip region with MediaPipe...")
                    lip_crops = []
                    annotated_frames = []
                    failed = 0

                    for frame in frames:
                        lip, annotated = crop_lip_region(frame)
                        if lip is not None and lip.size > 0:
                            lip_crops.append(lip)
                            annotated_frames.append(annotated)
                        else:
                            failed += 1

                    if not lip_crops:
                        st.error("No faces detected in the video. Please try a different clip with a clearly visible face.")
                        st.stop()

                    st.write(f"✅ Lip region detected in {len(lip_crops)}/{len(frames)} frames")

                    # Step 3 — Claude analysis
                    st.write("🧠 Sending lip frames to Claude Vision AI...")
                    prediction = analyze_lips_with_claude(lip_crops)
                    st.write("✅ Analysis complete!")
                    status.update(label="Analysis complete!", state="complete")

                st.session_state.prediction = prediction
                st.session_state.lip_crops = lip_crops[:8]
                st.session_state.annotated_frames = annotated_frames[:4]
                st.session_state.frame_count = len(frames)
                st.session_state.lip_count = len(lip_crops)
                st.rerun()

            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                os.unlink(tmp_path)

    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem 0; color:#333'>
            <div style='font-size:4rem'>🎬</div>
            <div style='margin-top:1rem; color:#555'>Upload a video to begin lip reading</div>
        </div>
        """, unsafe_allow_html=True)

with right:
    prediction = st.session_state.get("prediction", None)

    if not prediction:
        st.markdown("""
        <div style='text-align:center; padding:4rem 0; color:#333'>
            <div style='font-size:4rem'>👄</div>
            <div style='margin-top:1rem; color:#555'>Lip reading results will appear here</div>
            <div style='color:#444; font-size:0.85rem; margin-top:0.5rem'>Upload a video and click Read Lips</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Main prediction
        conf_color = {
            "High": "#22c55e",
            "Medium": "#f59e0b",
            "Low": "#ef4444"
        }.get(prediction.get("confidence", "Low"), "#888")

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-label'>👄 Predicted Speech</div>
            <div class='result-text'>"{prediction.get('predicted_text', '...')}"</div>
            <div style='margin-top:12px; color:{conf_color}; font-size:0.85rem; font-weight:600'>
                {prediction.get('confidence', '')} Confidence · {prediction.get('confidence_score', 0)}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value'>{st.session_state.get('frame_count', 0)}</div>
                <div class='stat-label'>Frames</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value'>{st.session_state.get('lip_count', 0)}</div>
                <div class='stat-label'>Lip Crops</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='stat-box'>
                <div class='stat-value' style='font-size:1.1rem'>{prediction.get('lip_activity', 'N/A')}</div>
                <div class='stat-label'>Lip Activity</div>
            </div>""", unsafe_allow_html=True)

        # Lip crops preview
        if st.session_state.get("lip_crops"):
            st.markdown("<div class='section-label' style='margin-top:20px'>🔬 Extracted Lip Frames</div>", unsafe_allow_html=True)
            cols = st.columns(len(st.session_state.lip_crops[:8]))
            for col, crop in zip(cols, st.session_state.lip_crops[:8]):
                with col:
                    st.image(crop, use_column_width=True)

        # Annotated frames
        if st.session_state.get("annotated_frames"):
            st.markdown("<div class='section-label' style='margin-top:16px'>📍 Lip Landmark Detection</div>", unsafe_allow_html=True)
            a_cols = st.columns(min(4, len(st.session_state.annotated_frames)))
            for col, frame in zip(a_cols, st.session_state.annotated_frames):
                with col:
                    st.image(frame, use_column_width=True)

        # Details
        st.markdown("<div class='section-label' style='margin-top:16px'>📊 Analysis Details</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='confidence-card'>
            <div style='color:#888; font-size:0.8rem; margin-bottom:6px'>👁️ <b>Observed:</b> {prediction.get('mouth_movements', '')}</div>
            <div style='color:#888; font-size:0.8rem; margin-bottom:6px'>🔤 <b>Phonemes:</b> {', '.join(prediction.get('phonemes_detected', []))}</div>
            <div style='color:#888; font-size:0.8rem'>📝 <b>Notes:</b> {prediction.get('notes', '')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Alternative predictions
        alts = prediction.get("alternative_predictions", [])
        if alts:
            st.markdown("<div class='section-label'>🔄 Alternative Predictions</div>", unsafe_allow_html=True)
            for alt in alts:
                st.markdown(f"""
                <div style='background:#111; border:1px solid #222; border-radius:8px; padding:10px 14px; margin-bottom:6px; color:#888; font-size:0.85rem'>
                    💬 "{alt}"
                </div>
                """, unsafe_allow_html=True)

        if st.button("🔄 Analyze Another Video", use_container_width=True):
            for key in ["prediction", "lip_crops", "annotated_frames", "frame_count", "lip_count"]:
                st.session_state.pop(key, None)
            st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center; color:#333; font-size:0.8rem'>Built with Python · MediaPipe · Claude Vision API · OpenCV · Streamlit</p>", unsafe_allow_html=True)
