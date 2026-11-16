import os
import json
import time
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import detector

# โหลดค่า Config จาก .env
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
CROPS_FOLDER = os.path.join(BASE_DIR, "static", "crops")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CROPS_FOLDER"] = CROPS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # สูงสุด 250MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """แสดงผลหน้าหลัก Dashboard"""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """ตรวจสอบสถานะของระบบ, Roboflow API และโมเดลจำแนก MobileNetV2"""
    roboflow_key = os.getenv("ROBOFLOW_API_KEY", "")
    has_valid_key = bool(roboflow_key and "your_" not in roboflow_key and "2He***" not in roboflow_key)
    
    return jsonify({
        "status": "online",
        "roboflow": {
            "configured": has_valid_key,
            "api_url": detector.ROBOFLOW_API_URL,
            "model_id": detector.ROBOFLOW_MODEL_ID,
            "client_active": detector.roboflow_client is not None
        },
        "classifier": {
            "name": "MobileNetV2",
            "type": detector.classifier_type,
            "model_path": detector.MODEL_PATH,
            "is_loaded": detector.classifier_model is not None,
            "classes": detector.CLASSES_RIPENESS
        },
        "defaults": {
            "frame_interval": detector.DEFAULT_FRAME_INTERVAL,
            "conf_threshold": detector.DEFAULT_CONF_THRESHOLD
        }
    })


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    """รับไฟล์วิดีโอจากผู้ใช้และบันทึกไว้ใน static/uploads"""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "ไม่พบไฟล์วิดีโอในคำขอ"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "ไม่ได้เลือกไฟล์"}), 400
        
    if file and allowed_file(file.filename):
        timestamp_prefix = int(time.time())
        safe_name = secure_filename(file.filename)
        filename = f"{timestamp_prefix}_{safe_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        
        video_url = f"/static/uploads/{filename}"
        return jsonify({
            "success": True,
            "filename": filename,
            "filepath": save_path,
            "url": video_url,
            "message": f"อัปโหลดไฟล์ {safe_name} สำเร็จ"
        })
    else:
        return jsonify({"success": False, "error": "รูปแบบไฟล์ไม่รองรับ (รองรับ MP4, AVI, MOV, WebM)"}), 400


@app.route("/api/process-video", methods=["POST"])
def process_video_batch():
    """ประมวลผลวิดีโอแบบรวดเดียว (Batch) และส่งผลลัพธ์ทั้งหมดกลับเป็น JSON"""
    data = request.get_json() or {}
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"success": False, "error": "กรุณาระบุชื่อไฟล์วิดีโอ"}), 400
        
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.isfile(video_path):
        return jsonify({"success": False, "error": f"ไม่พบไฟล์วิดีโอบนเซิร์ฟเวอร์: {filename}"}), 404
        
    frame_interval = int(data.get("frame_interval", detector.DEFAULT_FRAME_INTERVAL))
    conf_threshold = float(data.get("conf_threshold", detector.DEFAULT_CONF_THRESHOLD))
    
    detections = []
    generator = detector.process_video_generator(
        video_path=video_path,
        frame_interval=frame_interval,
        conf_threshold=conf_threshold,
        output_crops_dir=app.config["CROPS_FOLDER"]
    )
    
    total_detected = 0
    for event in generator:
        if event.get("type") == "detection":
            detections.append(event["data"])
            total_detected += 1
            
    # คำนวณสรุปสถิติ (4 คลาส)
    unripe = sum(1 for d in detections if d["stageKey"] == "Unripe")
    ripe = sum(1 for d in detections if d["stageKey"] == "Ripe")
    overripe = sum(1 for d in detections if d["stageKey"] == "Overripe")
    rotten = sum(1 for d in detections if d["stageKey"] == "Rotten")
    
    total = len(detections)
    stats = {
        "total": total,
        "unripe": unripe,
        "ripe": ripe,
        "overripe": overripe,
        "rotten": rotten,
        "pct_unripe": round((unripe / total * 100), 1) if total else 0,
        "pct_ripe": round((ripe / total * 100), 1) if total else 0,
        "pct_overripe": round((overripe / total * 100), 1) if total else 0,
        "pct_rotten": round((rotten / total * 100), 1) if total else 0
    }
    
    return jsonify({
        "success": True,
        "total_detected": total_detected,
        "stats": stats,
        "detections": detections
    })


@app.route("/api/process-video-stream")
def process_video_stream():
    """
    ประมวลผลวิดีโอแบบเรียลไทม์ผ่าน Server-Sent Events (SSE)
    หน้าเว็บจะได้รับข้อมูลการตรวจจับกล้วยทันทีที่ตรวจพบแต่ละเฟรม
    """
    filename = request.args.get("filename")
    if not filename:
        return Response("data: {\"type\": \"error\", \"message\": \"Missing filename\"}\n\n", mimetype="text/event-stream")
        
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.isfile(video_path):
        return Response(f"data: {{\"type\": \"error\", \"message\": \"Video not found: {filename}\"}}\n\n", mimetype="text/event-stream")
        
    frame_interval = int(request.args.get("frame_interval", detector.DEFAULT_FRAME_INTERVAL))
    conf_threshold = float(request.args.get("conf_threshold", detector.DEFAULT_CONF_THRESHOLD))

    def event_stream():
        generator = detector.process_video_generator(
            video_path=video_path,
            frame_interval=frame_interval,
            conf_threshold=conf_threshold,
            output_crops_dir=app.config["CROPS_FOLDER"]
        )
        for event in generator:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print("\n" + "=" * 60)
    print(f"🍌 BananaVision AI Server กำลังเริ่มต้น...")
    print(f"🌐 เปิดใช้งานที่: http://{host}:{port}")
    print(f"📂 โมเดล: {detector.MODEL_PATH} ({detector.classifier_type})")
    print("=" * 60 + "\n")
    
    app.run(host=host, port=port, debug=debug)
