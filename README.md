---
title: Banana Ripeness AI
emoji: 🍌
colorFrom: yellow
colorTo: amber
sdk: docker
app_port: 7860
pinned: false
---

# 🍌 BananaVision AI: ระบบตรวจจับและคัดแยกความสุกของกล้วยบนสายพาน (4 Classes)

ระบบตรวจจับพิกัดและจำแนกระดับความสุกของกล้วยอัตโนมัติจากคลิปวิดีโอบนสายพานลำเลียง (Conveyor Belt) ด้วยสถาปัตยกรรม **2-Stage Deep Learning Pipeline** พร้อมระบบแสดงผลบน **Web Application Dashboard** แบบเรียลไทม์ และระบบส่งออกรายงานสรุปข้อมูลในรูปแบบ CSV

---

## 📌 สรุปภาพรวมของโครงการ (Project Overview)

ระบบถูกออกแบบมาเพื่อจำลองและประยุกต์ใช้งานจริงในโรงงานคัดแยกผลผลิตทางการเกษตร โดยผสานเทคโนโลยี Computer Vision และ Deep Learning เข้าด้วยกัน:

1. **Stage 1 — Object Detection (YOLOv8 via Roboflow API):**
   * ตรวจจับตำแหน่งของกล้วยในแต่ละเฟรมวิดีโอบนสายพาน
   * ระบุพิกัด Bounding Box `[x, y, width, height]` พร้อมค่าความมั่นใจ (Confidence Score)
   * ตัดภาพเฉพาะส่วนของกล้วยแต่ละผล (Auto-Cropping) เพื่อส่งต่อให้สเตจถัดไป
2. **Stage 2 — Ripeness Classification (MobileNetV2):**
   * รับภาพกล้วยที่ตัดมาจาก Stage 1 ปรับขนาดสู่ `224x224 px`
   * ทำนายระดับความสุกด้วยโมเดล **MobileNetV2 (TensorFlow / Keras 100%)**
   * จำแนกผลลัพธ์ออกเป็น **4 ระดับ (4 Stages)** ได้แก่ **Unripe, Ripe, Overripe และ Rotten**
3. **Real-time Web Dashboard (Flask + SSE):**
   * แสดงผลวิดีโอพร้อมแถบความคืบหน้า (Progress Bar) ขณะประมวลผล
   * อัปเดตการตรวจจับแบบสดผ่าน **Server-Sent Events (SSE)**
   * สรุปสถิติ KPI จำนวนและสัดส่วนเปอร์เซ็นต์ของแต่ละระดับความสุก
   * แกลเลอรีภาพ Crop ที่ถูกตีกรอบ Bounding Box และป้าย Label
   * ตารางข้อมูลรายงานที่สามารถกรองดูตามระดับความสุก และดาวน์โหลดเป็นไฟล์ CSV (UTF-8 BOM สำหรับ Excel ภาษาไทย)

---

## 🏷️ รายละเอียดระดับความสุก 4 คลาส (4 Ripeness Stages)

โมเดลในโครงการนี้ได้รับการฝึกฝนเพื่อจำแนกกล้วยออกเป็น 4 ระดับอย่างแม่นยำ โดยจับคู่กับโครงสร้าง Dataset ดังนี้:

| ลำดับ | คลาส (Stage Key) | ชื่อภาษาไทย | รหัสสี & ธีม | ลักษณะทางกายภาพของผลกล้วย | การนำไปใช้งาน |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **`Unripe`** *(Class A)* | **ดิบ (Unripe)** | 🟢 เขียว (`#10b981`) | เปลือกสีเขียวสด เนื้อแน่น ยังไม่พร้อมรับประทาน | ขนส่งระยะไกล / รอการบ่ม |
| **2** | **`Ripe`** *(Class B)* | **สุกพอดี (Ripe)** | 🟡 เหลือง (`#f59e0b`) | เปลือกสีเหลืองทอง สุกพอดี รสชาติหวานอร่อย | วางจำหน่ายสด / บริโภคทันที |
| **3** | **`Overripe`** *(Class C)* | **สุกงอม (Overripe)** | 🟠 ส้ม (`#ea580c`) | สุกงอม เริ่มมีจุดกระสีน้ำตาล (Sugar Spots) | แปรรูปทำขนม (เค้กกล้วยหอม) |
| **4** | **`Rotten`** *(Class D)* | **เน่าเสีย (Rotten)** | 🔴 กุหลาบเข้ม (`#e11d48`) | เปลือกดำคล้ำ ช้ำ มีรอยเน่า ไม่สามารถบริโภคได้ | คัดแยกทิ้ง / ทำปุ๋ยอินทรีย์ |

---

## 🧠 ข้อมูลการฝึกฝนโมเดล (Model & Dataset Training)

* **Dataset:** ดึงข้อมูลผ่าน Kaggle API (`luischuquimarca/banana-ripeness`) รวม 2,795 ภาพ
  * **Train Set (80%):** 1,677 ภาพ
  * **Validation Set (20%):** 419 ภาพ
  * **Test Set:** 699 ภาพ
* **สถาปัตยกรรมโมเดล:**
  * Base Backbone: `MobileNetV2` (Pretrained บน ImageNet, Freeze Backbone)
  * Head Layers: `GlobalAveragePooling2D` &rarr; `Dropout(0.25)` &rarr; `Dense(4, activation='softmax')`
  * Optimizer: `Adam(learning_rate=1e-4)`, Loss: `sparse_categorical_crossentropy`
  * Callbacks: `EarlyStopping`, `ReduceLROnPlateau`
* **ความแม่นยำ (Test Accuracy):**
  * โมเดลผ่านการประเมินบน Test Set 699 ภาพ ได้ค่าความแม่นยำสูงเฉลี่ย **>90%** (โดยเฉพาะ Class Rotten ทำได้ถึง **95.27%**)
  * ตรวจสอบความโปร่งใสของโมเดลด้วย **Grad-CAM (Gradient-weighted Class Activation Mapping)** พบว่าโมเดลโฟกัสที่ผิวสัมผัส รอยกระ และรูปทรงกล้วยอย่างแม่นยำ ไม่สับสนกับฉากหลัง
* **ไฟล์น้ำหนักโมเดล:** จัดเก็บไว้ที่ `models/MobileNetV2_banana_model.keras` (ขนาด ~23.47 MB)

---

## 📁 โครงสร้างโปรเจกต์ (Project Directory Structure)

```text
BananaRipenessClassification/
├── .gitignore                               # กำหนดไฟล์ที่ไม่ให้นำขึ้น Git (เช่น .env, .venv)
├── readme.md                                # เอกสารสรุปโครงการและคู่มือการใช้งาน
├── BananaComVis_KaggleAPI_Colab.ipynb       # Jupyter Notebook เทรนโมเดลบน Google Colab
├── banana_ripeness_ai_dashboard.html        # ไฟล์ UI Dashboard ต้นแบบ (Frontend Mockup)
├── banana_ripeness_template.py              # สคริปต์ Template ตัวอย่าง Pipeline 2 สเตจ
├── models/
│   └── MobileNetV2_banana_model.keras       # ไฟล์น้ำหนักโมเดล MobileNetV2 (4 Classes)
└── BananaRipeness/                          # โฟลเดอร์ Web Application หลัก (Production)
    ├── app.py                               # Flask Web Server & API Endpoints
    ├── detector.py                          # AI Pipeline (YOLOv8 + MobileNetV2 Pure Inference)
    ├── run.py                               # สคริปต์สำหรับสั่งรันเซิร์ฟเวอร์
    ├── start.bat                            # สคริปต์เปิดรันแบบคลิกเดียวบน Windows
    ├── requirements.txt                     # รายการไลบรารี Dependencies ที่ต้องติดตั้ง
    ├── .env                                 # ไฟล์เก็บ API Key และการตั้งค่าส่วนบุคคล (ไม่ขึ้น Git)
    ├── .env.example                         # ไฟล์ตัวอย่างสำหรับการตั้งค่า
    ├── templates/
    │   └── index.html                       # หน้า Frontend UI Dashboard ที่เชื่อมต่อกับ Backend
    └── static/
        ├── uploads/                         # โฟลเดอร์จัดเก็บคลิปวิดีโอที่ผู้ใช้อัปโหลด
        └── crops/                           # โฟลเดอร์จัดเก็บภาพกล้วยที่ถูก Crop และตีกรอบ
```

---

## 🚀 คู่มือการติดตั้งและเริ่มใช้งาน (Quick Start Guide)

### 1. ความต้องการของระบบ (Prerequisites)
* ระบบปฏิบัติการ: Windows 10/11, macOS หรือ Linux
* Python 3.10 หรือ Python 3.11 (แนะนำ **Python 3.11**)
* เครื่องมือจัดการแพ็กเกจ: `pip` หรือ `uv` (แนะนำอย่างยิ่งเพื่อความเร็วในการติดตั้ง)

---

### 2. ตั้งค่าไฟล์สภาพแวดล้อม (`.env`)
เข้าไปที่โฟลเดอร์ `BananaRipeness`:
```bash
cd BananaRipeness
```
สร้างหรือแก้ไขไฟล์ `.env` (คัดลอกจาก `.env.example`) แล้วใส่ **Roboflow API Key** ของคุณ:
```env
# Roboflow Inference API Settings
ROBOFLOW_API_KEY=ใส่_Roboflow_API_Key_ของคุณที่นี่
ROBOFLOW_API_URL=https://detect.roboflow.com
ROBOFLOW_MODEL_ID=banana-detection-ieo3p/2

# Classification Model Path (MobileNetV2 4 Classes)
MODEL_PATH=../models/MobileNetV2_banana_model.keras

# Video Processing Parameters
DEFAULT_FRAME_INTERVAL=10
DEFAULT_CONF_THRESHOLD=0.75

# Web Application Settings
HOST=127.0.0.1
PORT=5000
DEBUG=True
```

---

### 3. การสร้าง Virtual Environment และติดตั้ง Dependencies

#### วิธีที่ 1: ติดตั้งผ่าน `uv` (แนะนำ - เร็วที่สุดภายใน 1 นาที)
```powershell
# 1. สร้าง Virtual Environment ด้วย Python 3.11
uv venv .venv --python 3.11

# 2. ติดตั้งแพ็กเกจทั้งหมดจาก requirements.txt
uv pip install -r requirements.txt --python .venv\Scripts\python.exe
```

#### วิธีที่ 2: ติดตั้งผ่าน `python venv` มาตรฐาน
```powershell
# 1. สร้าง Virtual Environment
python -m venv .venv

# 2. เปิดใช้งาน Virtual Environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. ติดตั้งแพ็กเกจ
pip install -r requirements.txt
```

---

### 4. สั่งรัน Web Application

รันคำสั่ง:
```powershell
.venv\Scripts\python.exe run.py
```
*(หรือเพียงดับเบิลคลิกที่ไฟล์ **`BananaRipeness\start.bat`**)*

เมื่อเซิร์ฟเวอร์เริ่มต้นสำเร็จ จะแสดงข้อความในเทอร์มินัล:
```text
[Detector] เชื่อมต่อ Roboflow Client สำเร็จ (Model: banana-detection-ieo3p/2)
[Detector] กำลังโหลดโมเดล MobileNetV2 จาก: .../models/MobileNetV2_banana_model.keras (ขนาด 23.47 MB)...
✅ [Detector] โหลดโมเดล MobileNetV2 (TensorFlow/Keras) สำเร็จ 100% พร้อมใช้งาน!
🚀 Server started at: http://127.0.0.1:5000
```

เปิดเว็บเบราว์เซอร์ไปที่:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 💻 คู่มือการใช้งานหน้าเว็บ (Web Application Usage)

1. **โหมดอัปโหลดวิดีโอจริง (Video Upload Mode - แนะนำ):**
   * ลากไฟล์วิดีโอสายพานกล้วย (`.mp4`, `.mov`, `.avi`, `.webm`) วางลงในช่อง Dropzone หรือคลิกเลือกไฟล์
   * ปรับตั้งค่า **Confidence Threshold** และ **Frame Interval** (ข้ามเฟรมเพื่อเร่งความเร็วในการประมวลผล) ที่การ์ดด้านขวา
   * กดปุ่ม **"เริ่มตรวจจับ (Start Inference)"**
   * ระบบจะแสดงแถบความคืบหน้า (Progress Bar) และแสดงภาพ Crop พร้อมป้ายสเตจและ Confidence Score ลงในแกลเลอรีสดทันทีที่ตรวจพบ
2. **โหมดจำลองสายพาน (Demo Conveyor Simulation):**
   * กดสลับแท็บไปที่ **"จำลองสายพาน (Demo)"** เพื่อทดสอบระบบหน้าบ้านโดยไม่ต้องอัปโหลดวิดีโอ
   * กล้วยจำลองทั้ง 4 สเตจจะเคลื่อนที่ผ่านเส้นตรวจจับ (Auto-Crop Line) เพื่อจำลองการตัดภาพและวิเคราะห์แบบ Interactive
3. **การกรองผลลัพธ์ (Filtering):**
   * เลือกตัวกรองที่ Dropdown ด้านบน เพื่อดูเฉพาะคลาสที่สนใจ: `Unripe`, `Ripe`, `Overripe`, หรือ `Rotten`
4. **การส่งออกรายงาน (Export CSV):**
   * กดปุ่ม **"ส่งออก CSV"** มุมขวาบน ระบบจะสร้างไฟล์รายงาน `banana_ripeness_report_YYYY-MM-DD.csv` ที่มีรหัสกล้วย, Timestamp, ระดับความสุก, ค่าความมั่นใจ, และพิกัด Bounding Box
   * รองรับภาษาไทยใน Microsoft Excel อย่างสมบูรณ์ผ่านการเข้ารหัส UTF-8 BOM

---

## 🔒 แนวทางความปลอดภัยและคำแนะนำสำหรับการ Git Push

ไฟล์ [`.gitignore`](file:///C:/Users/HP/Documents/GitHub/BananaRipenessClassification/.gitignore) ได้รับการตั้งค่าไว้เรียบร้อยแล้วเพื่อป้องกันความลับหลุด:

* ❌ **`.env`** จะถูกมองข้าม ไม่ถูก Push ขึ้น GitHub เด็ดขาด เพื่อความปลอดภัยของ API Key
* ❌ **`.venv/`** และ `__pycache__/` จะไม่ถูกส่งขึ้น Git
* ❌ ไฟล์วิดีโอใน `static/uploads/` และภาพ Crop ชั่วคราวใน `static/crops/` จะไม่ถูกส่งขึ้น Git
* ✅ ส่งเฉพาะไฟล์โค้ด, โมเดล, เอกสาร และ **`.env.example`** ขึ้น Git เท่านั้น

**คำสั่ง Push ขึ้น Git ที่ถูกต้อง:**
```bash
git status               # ตรวจสอบความปลอดภัย (ต้องไม่พบไฟล์ .env)
git add .
git commit -m "Complete BananaRipeness 4-class classification web application"
git push origin main
```

---

## ❓ คำถามที่พบบ่อย (Troubleshooting & FAQ)

* **Q: ถ้าเปิดไฟล์ `index.html` ขึ้นมาโดยตรง (เปิดแบบ `file:///...`) จะใช้งานได้ไหม?**
  * **A:** ดูหน้าตา UI และเล่นโหมดจำลองสายพาน (Demo) ได้ครับ แต่จะประมวลผลคลิปวิดีโอจริงไม่ได้ เนื่องจากโมเดล YOLOv8 และ MobileNetV2 ต้องทำงานผ่านเซิร์ฟเวอร์ Python หลังบ้านที่รันผ่าน `http://127.0.0.1:5000`
* **Q: เกิดข้อผิดพลาด `InferenceConfiguration.__init__() got an unexpected keyword argument 'api_key_transport'` เกิดจากอะไร?**
  * **A:** เกิดจากไลบรารี `inference-sdk` เวอร์ชันใหม่ได้ตัดพารามิเตอร์นี้ออก ในโค้ดปัจจุบันได้รับการแก้ไขให้ส่ง `api_url` และ `api_key` เข้า `InferenceHTTPClient(api_url=..., api_key=...)` โดยตรงแล้ว
* **Q: ทำไมต้องบังคับใช้ MobileNetV2 100% และไม่ใช้ Heuristic?**
  * **A:** เพราะโมเดล Deep Learning ที่เทรนมา 4 คลาสสามารถตรวจจับรอยช้ำ, ลวดลาย, และเฉดสีได้อย่างแม่นยำสูง (Accuracy >90%) การตัด Heuristic ออกช่วยรับประกันว่าผลลัพธ์ทุกชิ้นมาจากการตัดสินใจของ AI โมเดลจริง 100%
