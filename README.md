---
title: Banana Ripeness AI
emoji: 🍌
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# 🍌 Banana Ripeness Classification System

ระบบวิเคราะห์ความสุกของกล้วยแบบ Real-time โดยใช้ Computer Vision แบบ 2 ขั้นตอน (Two-Stage AI Pipeline) ประกอบด้วย Object Detection ด้วย YOLOv8 และ Image Classification ด้วย MobileNetV2

> **Live Demo:** [🤗 Hugging Face Spaces](https://huggingface.co/spaces/thananonny/BananaRipeness)

---

## 📋 สารบัญ

- [ภาพรวมระบบ](#ภาพรวมระบบ)
- [AI Pipeline Method](#ai-pipeline-method)
- [โครงสร้างโปรเจก](#โครงสร้างโปรเจก)
- [การติดตั้งและรันในเครื่อง](#การติดตั้งและรันในเครื่อง)
- [API Endpoints](#api-endpoints)
- [รายละเอียดโมเดล](#รายละเอียดโมเดล)
- [Deploy บน Hugging Face Spaces](#deploy-บน-hugging-face-spaces)

---

## ภาพรวมระบบ

ระบบสามารถรับ Input ได้ 3 รูปแบบ:

| Mode | Input | Output |
|------|-------|--------|
| 🎥 Video Upload | ไฟล์วิดีโอ (.mp4, .avi, .mov) | Streaming frame-by-frame พร้อม Bounding Box |
| 🖼️ Image Analysis | รูปภาพ 1 ใบ | ผลการตรวจจับและจำแนกชนิดความสุก |
| 🏭 Demo Conveyor Belt | ภาพตัวอย่างในระบบ | จำลองสายพานลำเลียงพร้อม animation |

**4 ระดับความสุกที่จำแนกได้:**

| ระดับ | สี Label | ความหมาย |
|-------|----------|-----------|
| 🟢 Fresh | เขียวสด | กล้วยสด/ดิบสด (Class A) |
| 🟢 Unripe | เขียวอมเหลือง | ยังดิบ/ห่าม (Class B) |
| 🟡 Ripe | เหลือง | สุกพอดี เหมาะสำหรับจำหน่าย (Class C) |
| 🟠 Overripe | ส้ม | สุกงอม เริ่มมีจุดน้ำตาล (Class D) |

---

## AI Pipeline Method

ระบบใช้สถาปัตยกรรม **Two-Stage Detection + Classification Pipeline**

```
Input (Video Frame / Image)
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Object Detection      │
│  YOLOv8s via Roboflow API       │
│  Model: banana-detection-ieo3p/2│
│  mAP: 85.78%                    │
│                                 │
│  → Detects banana locations     │
│  → Returns Bounding Boxes (x,y, │
│    w, h) + Banana Species class │
└─────────────┬───────────────────┘
              │  Crop each detected banana
              ▼
┌─────────────────────────────────┐
│  Stage 2: Ripeness Classifier   │
│  MobileNetV2 (Keras, 23.47 MB)  │
│  Input: 224×224 RGB [0, 255]    │
│                                 │
│  → 4-class Softmax output:      │
│     0: Fresh (Class A)          │
│     1: Unripe (Class B)         │
│     2: Ripe (Class C)           │
│     3: Overripe (Class D)       │
└─────────────┬───────────────────┘
              │
              ▼
   Annotated Output + Confidence %
```

### Stage 1 — YOLOv8s Object Detection (Roboflow)

- **Model:** `banana-detection-ieo3p/2` (YOLOv8s architecture)
- **Host:** Roboflow Cloud Inference API (`https://detect.roboflow.com`)
- **Input:** NumPy array (BGR frame จาก OpenCV)
- **Output:** รายการ Bounding Box พร้อม class ชนิดกล้วย 5 สายพันธุ์:
  - `kluai hom` (กล้วยหอม)
  - `kluai kai` (กล้วยไข่)
  - `Kluai Leb Mu Nang` (กล้วยเล็บมือนาง)
  - `kluai nam wa` (กล้วยน้ำว้า)
  - `kluai ta nee` (กล้วยตานี)

### Stage 2 — MobileNetV2 Ripeness Classifier

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Training Dataset:** [Banana Ripeness Dataset — Kaggle](https://www.kaggle.com/)
- **Input Size:** 224 × 224 × 3 (RGB)
- **Normalization:** Built-in ภายในโมเดล (`TrueDivide /127.5`, `Subtract -1.0`) — **ไม่ต้อง preprocess ซ้ำจากภายนอก**
- **Output:** Softmax 4 classes พร้อม Confidence score
- **File:** `models/MobileNetV2_banana_model.keras` (23.47 MB, tracked via Git LFS)

### Video Streaming (SSE)

สำหรับ mode วิดีโอ ระบบใช้ **Server-Sent Events (SSE)** เพื่อ stream ผลลัพธ์แบบ real-time:

```
Client                          Server (Flask)
  │                                   │
  │── POST /api/upload-video ─────────►│
  │                                   │  บันทึกไฟล์
  │◄─ { filename } ───────────────────│
  │                                   │
  │── GET /api/process-video-stream ──►│
  │                                   │  วนซ้ำทุก Frame
  │◄── data: { frame_b64, results } ──│  (YOLOv8 + MobileNetV2)
  │◄── data: { frame_b64, results } ──│
  │◄── data: [DONE] ──────────────────│
```

---

## โครงสร้างโปรเจก

```
BananaRipenessClassification/
├── Dockerfile                        # Docker config สำหรับ deploy บน HF Spaces
├── README.md                         # ไฟล์นี้
├── .gitattributes                    # Git LFS tracking สำหรับไฟล์ขนาดใหญ่
├── .dockerignore
│
├── BananaRipeness/                   # Flask Web Application
│   ├── app.py                        # Main Flask app + API Endpoints
│   ├── detector.py                   # AI Pipeline (Roboflow + MobileNetV2)
│   ├── run.py                        # Entry point สำหรับรัน local
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # ตัวอย่าง Environment Variables
│   ├── templates/
│   │   └── index.html                # Web Dashboard UI (3 modes)
│   └── static/
│       ├── demo_bananas/             # รูปตัวอย่างสำหรับ Demo mode
│       │   ├── unripe.png
│       │   ├── ripe.png
│       │   ├── overripe.png
│       │   ├── rotten.png
│       │   └── banana_bunch.jpg
│       ├── crops/                    # Cropped bounding boxes (temp)
│       └── uploads/                  # Uploaded videos (temp)
│
├── models/
│   └── MobileNetV2_banana_model.keras  # Trained model (23.47 MB, Git LFS)
│
└── BananaComVis_KaggleAPI_Colab.ipynb  # Notebook สำหรับ Training บน Kaggle/Colab
```

---

## การติดตั้งและรันในเครื่อง

### Prerequisites

- Python 3.10+
- [Git LFS](https://git-lfs.com/) (สำหรับดาวน์โหลด `.keras` model)
- Roboflow API Key ([สมัครฟรี](https://roboflow.com/))

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/BananaRipenessClassification.git
cd BananaRipenessClassification

# ดาวน์โหลดไฟล์ Git LFS (model .keras)
git lfs pull
```

### 2. ตั้งค่า Environment Variables

```bash
cp BananaRipeness/.env.example BananaRipeness/.env
```

แก้ไขไฟล์ `.env`:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_MODEL_ID=banana-detection-ieo3p/2
```

### 3. ติดตั้ง Dependencies

```bash
cd BananaRipeness
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. รัน Application

```bash
python run.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:7860`

### รันด้วย Docker

```bash
# Build image
docker build -t banana-ripeness .

# Run container
docker run -p 7860:7860 \
  -e ROBOFLOW_API_KEY=your_key \
  -e ROBOFLOW_MODEL_ID=banana-detection-ieo3p/2 \
  banana-ripeness
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web Dashboard UI |
| `GET` | `/api/status` | ตรวจสอบสถานะระบบและโมเดล |
| `POST` | `/api/upload-video` | อัปโหลดไฟล์วิดีโอ |
| `GET` | `/api/process-video-stream` | Stream ผลลัพธ์วิดีโอแบบ SSE |
| `POST` | `/api/process-demo-frame` | ประมวลผล Demo frame |
| `POST` | `/api/analyze-image` | วิเคราะห์รูปภาพนิ่ง |
| `GET` | `/api/demo-images` | ดึงรายการรูปตัวอย่าง |

### ตัวอย่าง: วิเคราะห์รูปภาพ

```bash
curl -X POST http://localhost:7860/api/analyze-image \
  -F "image=@banana.jpg"
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class": "kluai hom",
      "confidence": 0.92,
      "ripeness": "Ripe",
      "ripeness_confidence": 0.87,
      "bbox": { "x": 120, "y": 80, "width": 200, "height": 300 }
    }
  ],
  "total_bananas": 1,
  "annotated_image": "<base64 string>"
}
```

---

## รายละเอียดโมเดล

### Object Detection — YOLOv8s (Roboflow)

| Metric | Value |
|--------|-------|
| Architecture | YOLOv8s |
| mAP@0.5 | 85.78% |
| Classes | 5 ชนิดกล้วย |
| Host | Roboflow Cloud API |
| Model ID | `banana-detection-ieo3p/2` |

### Ripeness Classifier — MobileNetV2 (Keras)

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 (Transfer Learning) |
| Input Size | 224 × 224 × 3 |
| Output Classes | 4 (Fresh, Unripe, Ripe, Overripe) |
| Model Size | 23.47 MB |
| Format | Keras `.keras` |
| Storage | Git LFS |

---

## Deploy บน Hugging Face Spaces

Space นี้ Deploy ด้วย Docker SDK บน Hugging Face Spaces

### ขั้นตอน Deploy

```bash
# 1. เพิ่ม HF Space เป็น remote
git remote add space https://huggingface.co/spaces/<username>/banana-ripeness-ai

# 2. ตั้งค่า Secrets ใน HF Spaces Settings
#    ROBOFLOW_API_KEY = your_api_key

# 3. Push ขึ้น Space
git push space main
```

### Environment Variables ที่ต้องตั้งใน HF Spaces Secrets

| Variable | Description |
|----------|-------------|
| `ROBOFLOW_API_KEY` | Roboflow API Key สำหรับ YOLOv8 Inference |
| `ROBOFLOW_MODEL_ID` | Model ID (default: `banana-detection-ieo3p/2`) |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python, Flask |
| Object Detection | YOLOv8s via Roboflow Inference SDK |
| Classification | MobileNetV2 (TensorFlow / Keras) |
| Image Processing | OpenCV, PIL |
| Streaming | Server-Sent Events (SSE) |
| Frontend | HTML5, CSS3, JavaScript |
| Containerization | Docker |
| Deployment | Hugging Face Spaces (Docker SDK) |
| Model Storage | Git LFS |

---

## 📄 License

This project is for educational purposes.

---

*Built with ❤️ for Banana Quality Control*
