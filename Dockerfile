# ใช้ Python 3.11 Slim เป็น Base Image
FROM python:3.11-slim

# ปิด Buffering เพื่อให้เห็น Log ทันที และตั้งค่า Port ของ Hugging Face Spaces (7860)
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

# ติดตั้ง System Packages ที่ OpenCV ต้องการ (libgl, libglib) และ curl สำหรับดึงโมเดล
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ตั้ง Working Directory
WORKDIR /app

# คัดลอก requirements และติดตั้งโมดูล
COPY BananaRipeness/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโฟลเดอร์โมเดลและแอป
COPY models/ /app/models/
COPY BananaRipeness/ /app/BananaRipeness/

# ดาวน์โหลดโมเดลจริงขนาด 24.6 MB บรรจุเข้า Image ทันที (แก้ปัญหา Git LFS ข้ามทุกแพลตฟอร์ม 100%)
RUN curl -L -o /app/models/MobileNetV2_banana_model.keras \
    https://huggingface.co/spaces/thananonny/BananaRipeness/resolve/main/models/MobileNetV2_banana_model.keras

# สร้าง Non-root user (UID 1000) ตามข้อกำหนดความปลอดภัยของ Hugging Face Spaces
RUN useradd -m -u 1000 user && \
    mkdir -p /app/BananaRipeness/static/uploads /app/BananaRipeness/static/crops && \
    chown -R user:user /app

USER user
WORKDIR /app/BananaRipeness

# เปิด Port 7860 สำหรับ Hugging Face Spaces
EXPOSE 7860

# รัน Flask Application
CMD ["python", "app.py"]
