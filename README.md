<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=📸%20SnapClass%20AI%20Attendance&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Smart%20Attendance%20via%20Face%20%26%20Voice%20Recognition&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVC-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![dlib](https://img.shields.io/badge/dlib-Face%20Recognition-008080?style=for-the-badge)](http://dlib.net)
[![Resemblyzer](https://img.shields.io/badge/Resemblyzer-Voice%20ID-6C3483?style=for-the-badge)](https://github.com/resemble-ai/Resemblyzer)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

**SnapClass** is an AI-powered classroom attendance system that automates student identification using **face recognition** and **voice recognition** — no manual roll calls, no RFID cards, no overhead.

Teachers snap a photo of the class or play an audio clip, and SnapClass instantly marks attendance using two independent AI pipelines. Students can self-enroll via QR join codes, and all data is persisted in a Supabase backend.

| Feature | Details |
|:---|:---|
| 🎯 Task Type | Multi-modal Biometric Identification |
| 👁️ Face Model | dlib (HOG detector) + SVC classifier on 128-dim embeddings |
| 🎙️ Voice Model | Resemblyzer VoiceEncoder + cosine similarity matching |
| 🖥️ Frontend | Streamlit multi-screen app (Teacher / Student / Home) |
| 🗄️ Backend | Supabase (PostgreSQL + Storage) |
| 🔗 QR Enroll | Auto-enroll students via shareable join-code QR |

---

## 📂 Dataset / Enrollment

SnapClass does not use a pre-labeled public dataset — it builds its own biometric store from enrolled students:

- **Face enrollment:** Teachers add 1–N reference photos per student → dlib extracts 128-dim face descriptors → stored in Supabase
- **Voice enrollment:** Students record a voice sample → Resemblyzer embeds it → stored alongside face data
- **Training trigger:** SVC classifier is (re)trained on-demand using all enrolled face embeddings, with `class_weight='balanced'` to handle uneven enrollments

---

## 🔄 Pipeline Workflow

```
Photo/Audio Input → Preprocessing → Embedding Extraction → Model Inference → Threshold Filter → Attendance Record
```

### 1️⃣ Face Recognition Pipeline

```python
# dlib HOG detector → shape predictor → 128-dim face descriptor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(pose_predictor_model_location())
facerec = dlib.face_recognition_model_v1(face_recognition_model_location())

# SVC classifier trained on enrolled embeddings
clf = SVC(kernel='linear', probability=True, class_weight='balanced')
```

- Detects all faces in a group photo in one pass
- Generates a 128-dimensional embedding per detected face
- SVC predicts student ID; euclidean distance threshold (`≤ 0.6`) filters false positives
- Handles 1-class edge case (single enrolled student)

### 2️⃣ Voice Recognition Pipeline

```python
# Resemblyzer → d-vector embedding → cosine similarity
encoder = VoiceEncoder()
embedding = encoder.embed_utterance(preprocess_wav(audio))
similarity = np.dot(new_emb_normalized, stored_emb_normalized)
```

- Accepts single-speaker audio or bulk classroom recordings
- **Single-speaker:** computes cosine similarity against all enrolled voice embeddings; accepts if `similarity ≥ 0.82` AND margin over second-best `≥ 0.05`
- **Bulk audio:** librosa splits recording into speech segments (`top_db=25`) → each segment independently identified → deduplication keeps highest-confidence match per student

### 3️⃣ Attendance Aggregation

- Face & voice results merged per student ID
- Results surfaced in `dialog_attendance_results` UI component
- Records written to Supabase with timestamp, subject, and confidence source

---

## 🤖 Models

### 1️⃣ Face Identification — dlib + SVC ⭐ Primary Model

```python
SVC(kernel='linear', probability=True, class_weight='balanced')
# Trained on 128-dim dlib face descriptors
# Threshold: euclidean_distance <= 0.6
```

- dlib's `face_recognition_model_v1` produces ResNet-derived 128-dim embeddings
- Linear SVC is fast, interpretable, and works well with small per-class samples (real-world classroom sizes)
- Distance threshold prevents unknown faces from being misclassified

### 2️⃣ Voice Identification — Resemblyzer Cosine Similarity ⭐ Primary Model

```python
VoiceEncoder()  # GE2E-trained speaker embedding model
# Threshold: cosine_similarity >= 0.82
# Margin: best_score - second_best >= 0.05
```

- Resemblyzer uses a GE2E-trained LSTM to produce speaker d-vectors
- Dual threshold (absolute similarity + margin) drastically reduces false positives in noisy classrooms
- librosa-based segmentation enables bulk roll-call processing from a single audio file

---

## 📊 Recognition Thresholds

| Pipeline | Metric | Threshold | Notes |
|:---|:---|:---:|:---|
| Face Recognition | Euclidean Distance | `≤ 0.60` | Lower = stricter match |
| Voice Recognition | Cosine Similarity | `≥ 0.82` | Higher = stricter match |
| Voice Recognition | Confidence Margin | `≥ 0.05` | Gap over 2nd best match |
| Bulk Audio | Segment Length | `≥ 1.0s` | Short clips ignored |

---

## 🔍 Key Insights

- 🎯 **Dual-threshold voice matching** (absolute + margin) is critical — cosine similarity alone causes frequent misidentification in classrooms with similar-sounding voices
- 📸 **Group photo attendance** works because dlib's HOG detector processes all faces in one pass, making it O(faces) not O(students×faces)
- ⚖️ **`class_weight='balanced'`** in SVC is essential — otherwise students enrolled with more photos dominate the classifier boundary
- 🔄 **On-demand retraining** (`st.cache_resource.clear()`) keeps the model fresh as new students enroll without requiring a server restart
- 🎙️ **librosa `effects.split`** with `top_db=25` reliably isolates individual speech segments from classroom background noise for bulk roll-call

---

## 🗂️ Repository Structure

```
SnapClass-Ai-Attendance/
│
├── app.py                          # Streamlit entry point, screen router
├── requirements.txt                # All dependencies
├── LICENSE
│
└── src/
    ├── pipelines/
    │   ├── face_pipeline.py        # dlib + SVC face identification
    │   └── voice_pipeline.py       # Resemblyzer voice identification
    │
    ├── screens/
    │   ├── home_screen.py          # Login / role selection
    │   ├── teacher_screen.py       # Subject management, attendance capture
    │   └── student_screen.py       # Student dashboard, enrollment
    │
    ├── components/
    │   ├── dialog_add_photo.py     # Face enrollment dialog
    │   ├── dialog_attendance_results.py  # Results viewer
    │   ├── dialog_auto_enroll.py   # QR join-code enrollment
    │   ├── dialog_create_subject.py
    │   ├── dialog_enroll.py
    │   ├── dialog_share_subject.py # QR code generator (segno)
    │   ├── dialog_voice_attendance.py
    │   ├── header.py
    │   ├── footer.py
    │   └── subject_card.py
    │
    ├── database/
    │   ├── config.py               # Supabase credentials
    │   └── db.py                   # DB read/write helpers
    │
    └── ui/
        └── base_layout.py          # Shared layout wrapper
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ronakrajput8882/SnapClass-Ai-Attendance.git
cd SnapClass-Ai-Attendance

# 2. Install dependencies (dlib-bin avoids manual cmake build)
pip install -r requirements.txt

# 3. Set up Supabase credentials in src/database/config.py
#    SUPABASE_URL = "your-project-url"
#    SUPABASE_KEY = "your-anon-key"

# 4. Run the app
streamlit run app.py
```

> **Note:** `dlib-bin` is used instead of `dlib` to skip the C++ build step. Requires Python 3.10+.

---

## 🧠 Key Learnings

- Building a biometric attendance system end-to-end reveals how much the **enrollment quality** (lighting, angle, audio clarity) matters more than model choice
- Streamlit's `st.cache_resource` is a practical solution for loading heavy ML models (dlib, Resemblyzer) once per session
- Supabase's real-time capabilities + Python SDK make it an ideal lightweight backend for prototyping production-grade ML apps
- Voice identification in group settings requires **margin-based thresholding**, not just absolute similarity — cosine similarity alone is insufficient
- QR-based auto-enrollment dramatically reduces the friction of onboarding students to biometric systems

---

## 🛠️ Tech Stack

| Tool | Purpose |
|:---|:---|
| Python 3.10+ | Core language |
| Streamlit | Web app framework |
| dlib + face_recognition_models | Face detection & 128-dim embedding |
| scikit-learn (SVC) | Face classification |
| Resemblyzer | Speaker voice embedding (GE2E) |
| librosa | Audio loading & speech segmentation |
| Supabase | Cloud database & storage backend |
| segno | QR code generation for join links |
| Pillow | Image processing |
| NumPy / Pandas | Numerical ops & data handling |
| bcrypt | Password hashing |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>
