import os
import json
import time
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# โหลดค่า Config จาก .env
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH, override=True)

import detector

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
CROPS_FOLDER = os.path.join(BASE_DIR, "static", "crops")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CROPS_FOLDER"] = CROPS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # สูงสุด 250MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


@app.route("/")
def index():
    """แสดงผลหน้าหลัก Dashboard"""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """ตรวจสอบสถานะของระบบ, Roboflow API และโมเดลจำแนก MobileNetV2"""
    roboflow_key = os.getenv("ROBOFLOW_API_KEY", "")
    has_valid_key = bool(roboflow_key and "your_" not in roboflow_key)
    api_url = os.getenv("ROBOFLOW_API_URL", detector.ROBOFLOW_API_URL)
    model_id = os.getenv("ROBOFLOW_MODEL_ID", detector.ROBOFLOW_MODEL_ID)
    
    return jsonify({
        "status": "online",
        "roboflow": {
            "configured": has_valid_key,
            "detector_name": "YOLOv8s",
            "api_url": api_url,
            "model_id": model_id,
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


@app.route("/api/process-demo-frame", methods=["POST"])
def process_demo_frame():
    """
    รับภาพกล้วยจากหน้า Demo (Conveyor Simulation):
    1. ส่งภาพเข้า Roboflow สำหรับ Object Detection & Bounding Box จริง
    2. Crop ภาพตาม Bounding Box จาก Roboflow
    3. ส่งเข้า MobileNetV2 สำหรับการจำแนกความสุก 4 ระดับ (Unripe, Ripe, Overripe, Rotten)
    4. วาด Annotation และบันทึกภาพลง static/crops/
    """
    data = request.get_json() or {}
    image_data = data.get("image")
    banana_id = int(data.get("id", 1))
    timestamp_sec = float(data.get("timestamp_sec", 0.0))
    conf_threshold = float(data.get("conf_threshold", 0.35))
    
    if not image_data:
        return jsonify({"success": False, "error": "ไม่พบข้อมูลภาพ base64"}), 400
        
    try:
        # ถอดรหัส base64 image data URL (เช่น data:image/jpeg;base64,...)
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
            
        img_bytes = base64.b64decode(image_data)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr_frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if bgr_frame is None or bgr_frame.size == 0:
            return jsonify({"success": False, "error": "ไม่สามารถแปลงข้อมูลภาพได้"}), 400
            
        record = detector.process_demo_banana(
            bgr_snapshot=bgr_frame,
            banana_counter=banana_id,
            conf_threshold=conf_threshold,
            output_crops_dir=app.config["CROPS_FOLDER"],
            timestamp_sec=timestamp_sec
        )
        
        return jsonify({
            "success": True,
            "detection": record
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"การประมวลผลผิดพลาด: {str(e)}"}), 500


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
            
    # คำนวณสรุปสถิติ (4 คลาส: Fresh, Unripe, Ripe, Overripe)
    fresh = sum(1 for d in detections if d["stageKey"] in ("Fresh", "fresh"))
    unripe = sum(1 for d in detections if d["stageKey"] in ("Unripe", "unripe"))
    ripe = sum(1 for d in detections if d["stageKey"] in ("Ripe", "ripe"))
    overripe = sum(1 for d in detections if d["stageKey"] in ("Overripe", "overripe", "Rotten", "rotten"))
    
    total = len(detections)
    stats = {
        "total": total,
        "fresh": fresh,
        "unripe": unripe,
        "ripe": ripe,
        "overripe": overripe,
        "rotten": overripe,  # คงคีย์ไว้สำหรับ backward compatibility
        "pct_fresh": round((fresh / total * 100), 1) if total else 0,
        "pct_unripe": round((unripe / total * 100), 1) if total else 0,
        "pct_ripe": round((ripe / total * 100), 1) if total else 0,
        "pct_overripe": round((overripe / total * 100), 1) if total else 0,
        "pct_rotten": round((overripe / total * 100), 1) if total else 0
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


@app.route("/api/demo-images")
def api_demo_images():
    """ส่งรายการภาพ Demo สำหรับทดลองใช้งาน (ไม่ซ้ำซ้อน และอิงตามไฟล์จริงใน static/demo_bananas)"""
    demo_dir = os.path.join(BASE_DIR, "static", "demo_bananas")
    priority_items = [
        {"filename": "fresh.png", "label": "สด (Fresh)", "badge": "bg-emerald-100 text-emerald-800"},
        {"filename": "ripe.png", "label": "สุกพอดี (Ripe)", "badge": "bg-amber-100 text-amber-800"},
        {"filename": "overripe.png", "label": "สุกงอม (Overripe)", "badge": "bg-orange-100 text-orange-800"},
        {"filename": "rotten.png", "label": "สุกงอมมาก (Overripe)", "badge": "bg-rose-100 text-rose-800"},
        {"filename": "banana_brunch.png", "label": "ภาพหวีกล้วยจริง (Multi)", "badge": "bg-sky-100 text-sky-800"},
    ]
    images = []
    seen_files = set()
    if os.path.isdir(demo_dir):
        for item in priority_items:
            fpath = os.path.join(demo_dir, item["filename"])
            if os.path.isfile(fpath) and item["filename"] not in seen_files:
                seen_files.add(item["filename"])
                images.append({
                    "filename": item["filename"],
                    "url": f"/static/demo_bananas/{item['filename']}",
                    "label": item["label"],
                    "badge": item["badge"]
                })
    return jsonify({"success": True, "images": images})


@app.route("/api/analyze-image", methods=["POST"])
def analyze_image():
    """
    วิเคราะห์ภาพนิ่ง 1 ใบแบบ Full Pipeline:
    - รับ multipart/form-data (field: image) หรือ JSON base64 หรือ demo_filename
    - Stage 1: Roboflow detect → Bounding Boxes
    - Stage 2: Crop แต่ละ BBox → MobileNetV2 classify 4 คลาส
    - คืน JSON: detections[], annotated_image (base64 JPEG มี bbox วาดไว้)
    """
    conf_threshold = float(request.form.get("conf_threshold",
                           request.args.get("conf_threshold", detector.DEFAULT_CONF_THRESHOLD)))
    bgr_image = None

    # รับจากชื่อไฟล์ demo ใน demo_bananas
    req_json = request.get_json(silent=True) or {}
    demo_filename = request.form.get("demo_filename") or req_json.get("demo_filename")
    if demo_filename:
        safe_demo = os.path.basename(demo_filename)
        demo_path = os.path.join(BASE_DIR, "static", "demo_bananas", safe_demo)
        if os.path.isfile(demo_path):
            with open(demo_path, "rb") as f:
                bgr_image = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

    # รับจาก multipart file upload
    if bgr_image is None and "image" in request.files:
        file = request.files["image"]
        if file and file.filename and allowed_image(file.filename):
            file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # รับจาก JSON base64
    if bgr_image is None:
        image_data = req_json.get("image", "")
        conf_threshold = float(req_json.get("conf_threshold", conf_threshold))
        if image_data:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            img_bytes = __import__("base64").b64decode(image_data)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if bgr_image is None or bgr_image.size == 0:
        return jsonify({"success": False, "error": "ไม่พบข้อมูลภาพหรือรูปแบบไม่รองรับ"}), 400

    try:
        result = detector.analyze_image_full(
            bgr_image=bgr_image,
            conf_threshold=conf_threshold,
            output_crops_dir=app.config["CROPS_FOLDER"],
        )
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": f"การประมวลผลผิดพลาด: {str(e)}"}), 500


if __name__ == "__main__":
    is_cloud = bool(os.getenv("SPACE_ID") or os.getenv("RENDER") or os.getenv("PORT"))
    default_host = "0.0.0.0" if is_cloud else "127.0.0.1"
    default_port = 7860 if os.getenv("SPACE_ID") else 5000
    host = os.getenv("HOST", default_host)
    port = int(os.getenv("PORT", default_port))
    debug = os.getenv("DEBUG", "False" if is_cloud else "True").lower() == "true"
    
    print("\n" + "=" * 60)
    print(f"🍌 BananaVision AI Server กำลังเริ่มต้น...")
    print(f"🌐 เปิดใช้งานที่: http://{host}:{port}")
    print(f"📂 โมเดล: {detector.MODEL_PATH} ({detector.classifier_type})")
    print("=" * 60 + "\n")
    
    app.run(host=host, port=port, debug=debug)
