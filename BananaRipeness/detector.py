import os
import cv2
import time
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# โหลดค่าตัวแปรจาก .env (ค้นหา path ที่แน่นอน)
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH, override=True)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "banana-detection-ieo3p/2")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/MobileNetV2_banana_model.keras")
DEFAULT_FRAME_INTERVAL = int(os.getenv("DEFAULT_FRAME_INTERVAL", 10))
DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF_THRESHOLD", 0.75))

# รายละเอียด Stage และการจัดหมวดหมู่สี (4 คลาส: Fresh, Unripe, Ripe, Overripe)
STAGES_INFO = {
    "Fresh": {
        "key": "Fresh",
        "labelTh": "สด (Fresh)",
        "badgeClass": "bg-emerald-100 text-emerald-800 border-emerald-300",
        "hexColor": "#10b981",
        "bgrColor": (129, 185, 16)
    },
    "Unripe": {
        "key": "Unripe",
        "labelTh": "ดิบ (Unripe)",
        "badgeClass": "bg-teal-100 text-teal-800 border-teal-300",
        "hexColor": "#14b8a6",
        "bgrColor": (166, 184, 20)
    },
    "Ripe": {
        "key": "Ripe",
        "labelTh": "สุกพอดี (Ripe)",
        "badgeClass": "bg-amber-100 text-amber-800 border-amber-300",
        "hexColor": "#f59e0b",
        "bgrColor": (11, 158, 245)
    },
    "Overripe": {
        "key": "Overripe",
        "labelTh": "สุกงอม (Overripe)",
        "badgeClass": "bg-orange-100 text-orange-800 border-orange-300",
        "hexColor": "#ea580c",
        "bgrColor": (12, 88, 234)
    }
}

# เพิ่ม alias สำหรับตัวพิมพ์เล็ก และ Rotten เก่า เพื่อความเข้ากันได้
STAGES_INFO["fresh"] = STAGES_INFO["Fresh"]
STAGES_INFO["unripe"] = STAGES_INFO["Unripe"]
STAGES_INFO["ripe"] = STAGES_INFO["Ripe"]
STAGES_INFO["overripe"] = STAGES_INFO["Overripe"]
STAGES_INFO["Rotten"] = {
    "key": "Overripe",
    "labelTh": "สุกงอม (Overripe)",
    "badgeClass": "bg-rose-100 text-rose-800 border-rose-300",
    "hexColor": "#e11d48",
    "bgrColor": (72, 29, 225)
}
STAGES_INFO["rotten"] = STAGES_INFO["Rotten"]

# ลำดับ 4 คลาสตามโมเดล MobileNetV2 ที่เทรนไว้ (Class A -> D: Fresh, Unripe, Ripe, Overripe)
CLASSES_RIPENESS = ["Fresh", "Unripe", "Ripe", "Overripe"]

# Global Objects
roboflow_client = None
classifier_model = None
classifier_type = None  # 'keras' or 'torch'


def init_roboflow_client():
    """ตั้งค่า Roboflow Client สำหรับ YOLOv8 Detection"""
    global roboflow_client
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key or "your_" in api_key:
        print("[Detector] Roboflow API Key ยังไม่ได้ตั้งค่าหรือเป็นค่าตัวอย่างใน .env")
        roboflow_client = None
        return False
    try:
        from inference_sdk import InferenceHTTPClient
        api_url = os.getenv("ROBOFLOW_API_URL", "https://detect.roboflow.com")
        model_id = os.getenv("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID)
        roboflow_client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key
        )
        roboflow_client.use_api_v0()
        print(f"[Detector] เชื่อมต่อ Roboflow Client สำเร็จ (Model: {model_id}, URL: {api_url})")
        return True
    except Exception as e:
        print(f"[Detector] ไม่สามารถเชื่อมต่อ Roboflow Client ได้: {e}")
        roboflow_client = None
        return False


def resolve_model_path(path_str):
    """ค้นหาไฟล์โมเดล MobileNetV2 จาก path ต่างๆ"""
    candidates = [
        path_str,
        os.path.join(os.path.dirname(__file__), path_str),
        os.path.join(os.path.dirname(__file__), "..", "models", "MobileNetV2_banana_model.keras"),
        os.path.join(os.path.dirname(__file__), "models", "MobileNetV2_banana_model.keras"),
        os.path.join(os.path.dirname(__file__), "ripeness_model.pth"),
        os.path.join(os.path.dirname(__file__), "..", "ripeness_model.pth"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 1000:
            return os.path.abspath(c)
    return None


def init_classifier_model():
    """
    โหลดโมเดล MobileNetV2 สำหรับจำแนก 4 คลาส (บังคับใช้โมเดล 100% ไม่มี Heuristic)
    """
    global classifier_model, classifier_type
    model_file = resolve_model_path(MODEL_PATH)
    
    if not model_file:
        raise FileNotFoundError(
            f"❌ [Detector] ไม่พบไฟล์โมเดล MobileNetV2 ที่ระบุ ({MODEL_PATH})\n"
            f"กรุณาตรวจสอบว่ามีไฟล์ MobileNetV2_banana_model.keras ในโฟลเดอร์ models/"
        )

    print(f"[Detector] กำลังโหลดโมเดล MobileNetV2 จาก: {model_file} (ขนาด {os.path.getsize(model_file)/(1024*1024):.2f} MB)...")
    
    # โหลดโมเดล Keras (.keras / .h5)
    if model_file.endswith(".keras") or model_file.endswith(".h5"):
        try:
            import tensorflow as tf
            classifier_model = tf.keras.models.load_model(model_file, compile=False)
            classifier_type = "keras"
            print("✅ [Detector] โหลดโมเดล MobileNetV2 (TensorFlow/Keras) สำเร็จ 100% พร้อมใช้งาน!")
            return
        except Exception as e:
            raise RuntimeError(f"❌ [Detector] ล้มเหลวในการโหลดโมเดล Keras จาก {model_file}: {e}")

    # กรณีไฟล์เป็น PyTorch (.pth)
    if model_file.endswith(".pth"):
        try:
            import torch
            classifier_model = torch.load(model_file, map_location="cpu")
            if hasattr(classifier_model, "eval"):
                classifier_model.eval()
            classifier_type = "torch"
            print("✅ [Detector] โหลดโมเดล MobileNetV2 (PyTorch) สำเร็จ 100% พร้อมใช้งาน!")
            return
        except Exception as e:
            raise RuntimeError(f"❌ [Detector] ล้มเหลวในการโหลดโมเดล PyTorch จาก {model_file}: {e}")

    raise ValueError(f"❌ [Detector] ไม่รองรับรูปแบบไฟล์โมเดล: {model_file}")


def classify_crop(bgr_crop):
    """
    ส่งภาพ Crop เข้า MobileNetV2 เพื่อทำนายระดับความสุก 4 Stage
    (Fresh, Unripe, Ripe, Overripe) พร้อมค่า Confidence
    """
    global classifier_model, classifier_type
    
    if classifier_model is None:
        raise RuntimeError("❌ [Detector] โมเดล MobileNetV2 ยังไม่ได้ถูกโหลดเข้าหน่วยความจำ!")

    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("❌ [Detector] ภาพ Crop ว่างเปล่า ไม่สามารถส่งเข้าโมเดลได้")

    # โมเดล TensorFlow/Keras
    if classifier_type == "keras":
        # แปลง BGR -> RGB และปรับขนาดเป็น 224x224 ตาม Input Shape ของ MobileNetV2
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((224, 224))
        arr = np.array(img, dtype=np.float32)

        # หมายเหตุสำคัญ: โมเดล MobileNetV2 ตัวนี้มีเลเยอร์ true_divide (/127.5) และ subtract (-1.0)
        # ฝังอยู่ภายในตัวโมเดลแล้ว จึงต้องส่งค่าพิกเซล RGB ในช่วง [0, 255] เข้าไปโดยตรง
        # ห้ามเรียก preprocess_input ซ้ำซ้อนภายนอก เพราะจะทำให้ภาพถูก normalize 2 รอบจนกลายเป็นค่า -1 ทั้งหมด
        arr = np.expand_dims(arr, axis=0)
        
        preds = classifier_model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        
        # แมป Class ID สู่ 4 Stages:
        # Index 0: Class A (Fresh), 1: Class B (Unripe), 2: Class C (Ripe), 3: Class D (Overripe)
        stage_key = CLASSES_RIPENESS[pred_idx]
        return stage_key, confidence

    # โมเดล PyTorch
    if classifier_type == "torch":
        import torch
        from torchvision import transforms
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = preprocess(pil_img).unsqueeze(0)
        with torch.no_grad():
            out = classifier_model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx].item())
        stage_key = CLASSES_RIPENESS[pred_idx]
        return stage_key, confidence

    raise RuntimeError(f"❌ [Detector] ไม่พบ classifier_type ที่ถูกต้อง ({classifier_type})")


def fallback_banana_detection(frame):
    """
    ตรวจจับกล้วยด้วย OpenCV (Color Segmentation + Contour) 
    ใช้เฉพาะเมื่อ Roboflow ไม่ตอบกลับหรือไม่มี API Key สำหรับ Stage 1
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 30, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions = []
    min_area = (frame.shape[0] * frame.shape[1]) * 0.005
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio >= 1.1:
                predictions.append({
                    "x": x + w / 2,
                    "y": y + h / 2,
                    "width": w,
                    "height": h,
                    "confidence": min(0.92, 0.70 + (area / (frame.shape[0] * frame.shape[1])) * 2),
                    "class": "banana"
                })
    return predictions


def detect_bananas_in_frame(frame, temp_image_path=None):
    """
    เรียกใช้ Roboflow เพื่อหาตำแหน่งกล้วยในภาพ (รองรับ numpy array BGR โดยตรง)
    """
    global roboflow_client
    model_id = os.getenv("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID)
    if roboflow_client is not None:
        try:
            # inference_sdk รองรับ numpy array ได้โดยตรง
            res = roboflow_client.infer(frame, model_id=model_id)
            return res.get("predictions", [])
        except Exception as e:
            print(f"[Detector] Roboflow infer ล้มเหลว ({e}) - สลับใช้ Fallback Detection")
    
    return fallback_banana_detection(frame)


def annotate_crop_image(cropped_banana, stage_info, banana_id, conf_pct):
    """
    วาด Bounding Box และป้ายชื่อสเตจของโมเดล MobileNetV2 ลงบนภาพ Crop
    """
    annotated = cropped_banana.copy()
    h, w = annotated.shape[:2]
    color_bgr = stage_info["bgrColor"]
    
    cv2.rectangle(annotated, (2, 2), (w - 2, h - 2), color_bgr, 3)
    
    label_text = f"#{banana_id:02d} {stage_info['key']} {int(conf_pct)}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
    
    tag_h = th + baseline + 8
    cv2.rectangle(annotated, (2, 2), (min(w - 2, tw + 12), tag_h), color_bgr, -1)
    cv2.putText(annotated, label_text, (6, th + 5), font, font_scale, (15, 23, 42), thickness, cv2.LINE_AA)
    
    return annotated


def crop_to_base64(bgr_image, quality=85):
    """
    แปลงภาพ BGR เป็น Data URL (Base64 JPEG) ในหน่วยความจำ (RAM)
    เพื่อส่งให้ Browser นำไปแคชและแสดงผลได้ทันที โดยไม่ต้องเขียนไฟล์ลงดิสก์
    """
    if bgr_image is None or bgr_image.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("utf-8")


def process_demo_banana(bgr_snapshot, banana_counter=1, conf_threshold=0.35, output_crops_dir=None, timestamp_sec=0.0):
    """
    ประมวลผลกล้วยในโหมดสายพานจำลอง (Demo):
    1. ส่งภาพเข้า Roboflow หา Bounding Box จริง
    2. ทำการ Crop ภาพตาม Bounding Box ของ Roboflow
    3. ส่งภาพ Crop เข้าโมเดล MobileNetV2 จำแนก 4 ระดับความสุกจริง (Fresh, Unripe, Ripe, Overripe)
    4. วาด Bounding Box & Annotation ลงบนภาพ Crop แปลงเป็น Base64 Data URL แคชบน Browser
    5. ส่งข้อมูลผลลัพธ์กลับไปยังหน้าบ้าน (Zero-disk / ไม่บันทึกลงดิสก์ static/crops)
    """
    if output_crops_dir:
        os.makedirs(output_crops_dir, exist_ok=True)
    
    h_img, w_img = bgr_snapshot.shape[:2]
    
    # 1. รัน Roboflow Detection
    predictions = detect_bananas_in_frame(bgr_snapshot)
    
    roboflow_info = {
        "detected": False,
        "class": "banana",
        "confidence": 0.0,
        "box": None
    }
    
    target_pred = None
    if predictions:
        sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
        if sorted_preds[0].get("confidence", 0) >= conf_threshold:
            target_pred = sorted_preds[0]
        elif len(sorted_preds) > 0 and sorted_preds[0].get("confidence", 0) >= 0.20:
            target_pred = sorted_preds[0]

    if target_pred:
        bx = float(target_pred["x"])
        by = float(target_pred["y"])
        bw = float(target_pred["width"])
        bh = float(target_pred["height"])
        bconf = float(target_pred.get("confidence", 0.0))
        bcls = str(target_pred.get("class", "banana"))
        
        roboflow_info = {
            "detected": True,
            "class": bcls,
            "confidence": round(bconf * 100, 1),
            "box": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)]
        }
        
        x1 = max(0, int(bx - bw / 2))
        y1 = max(0, int(by - bh / 2))
        x2 = min(w_img, int(bx + bw / 2))
        y2 = min(h_img, int(by + bh / 2))
    else:
        # Fallback กรณี Roboflow ไม่พบ ให้ Crop โซนกลางภาพ
        pad_w = int(w_img * 0.1)
        pad_h = int(h_img * 0.1)
        x1, y1 = pad_w, pad_h
        x2, y2 = w_img - pad_w, h_img - pad_h
        bx = w_img / 2.0
        by = h_img / 2.0
        bw = float(x2 - x1)
        bh = float(y2 - y1)
    
    cropped = bgr_snapshot[y1:y2, x1:x2]
    if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
        cropped = bgr_snapshot
    
    # 2. ส่งภาพ Crop เข้า MobileNetV2 (100% Real Deep Learning Classification)
    stage_key, stage_conf = classify_crop(cropped)
    stage_info = STAGES_INFO.get(stage_key, STAGES_INFO["Ripe"])
    conf_pct = round(stage_conf * 100, 1)
    
    # 3. วาด Bounding Box & แปลงภาพเป็น Base64 Data URL ใน RAM
    annotated_crop = annotate_crop_image(cropped, stage_info, banana_counter, conf_pct)
    crop_image_url = crop_to_base64(annotated_crop)
    if output_crops_dir:
        crop_filename = f"crop_demo_{banana_counter:04d}_{int(time.time() * 1000) % 100000}.jpg"
        crop_path = os.path.join(output_crops_dir, crop_filename)
        cv2.imwrite(crop_path, annotated_crop)
        crop_image_url = f"/static/crops/{crop_filename}"
    
    return {
        "id": f"BN-{banana_counter:04d}",
        "timestamp": f"{timestamp_sec:.1f}s",
        "stageKey": stage_info["key"],
        "stageLabel": stage_info["labelTh"],
        "badgeClass": stage_info["badgeClass"],
        "confidence": conf_pct,
        "roboflow": roboflow_info,
        "bbox": f"[{int(x1)}, {int(y1)}, {int(bw)}, {int(bh)}]",
        "box_coords": {"x": int(x1), "y": int(y1), "w": int(bw), "h": int(bh)},
        "image": crop_image_url
    }


def process_video_generator(video_path, frame_interval=DEFAULT_FRAME_INTERVAL, conf_threshold=DEFAULT_CONF_THRESHOLD, output_crops_dir=None):
    """
    ประมวลผลวิดีโอแบบ Generator ส่งผลลัพธ์ผ่าน Roboflow + MobileNetV2 100%
    """
    if output_crops_dir:
        os.makedirs(output_crops_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield {"type": "error", "message": f"ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}"}
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    duration_sec = total_frames / fps
    
    yield {
        "type": "start",
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 1)
    }
    
    frame_count = 0
    banana_counter = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % frame_interval != 0:
                continue
                
            elapsed_sec = frame_count / fps
            progress_pct = min(100, int((frame_count / total_frames) * 100))
            
            # Stage 1: ตรวจจับตำแหน่งด้วย Roboflow
            predictions = detect_bananas_in_frame(frame)
            
            for pred in predictions:
                conf = pred.get("confidence", 0.0)
                if conf < conf_threshold:
                    continue
                    
                x = pred["x"]
                y = pred["y"]
                w = pred["width"]
                h = pred["height"]
                
                x1 = max(0, int(x - w / 2))
                y1 = max(0, int(y - h / 2))
                x2 = min(frame.shape[1], int(x + w / 2))
                y2 = min(frame.shape[0], int(y + h / 2))
                
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    continue
                    
                banana_counter += 1
                
                # Stage 2: ส่งเข้า MobileNetV2 100%
                stage_key, stage_conf = classify_crop(cropped)
                stage_info = STAGES_INFO.get(stage_key, STAGES_INFO["Ripe"])
                conf_pct = round(stage_conf * 100, 1)
                
                annotated_crop = annotate_crop_image(cropped, stage_info, banana_counter, conf_pct)
                crop_image_url = crop_to_base64(annotated_crop)
                if output_crops_dir:
                    crop_filename = f"crop_bn_{banana_counter:04d}_{frame_count}.jpg"
                    crop_path = os.path.join(output_crops_dir, crop_filename)
                    cv2.imwrite(crop_path, annotated_crop)
                    crop_image_url = f"/static/crops/{crop_filename}"
                
                detection_record = {
                    "id": f"BN-{banana_counter:04d}",
                    "timestamp": f"{elapsed_sec:.1f}s",
                    "stageKey": stage_info["key"],
                    "stageLabel": stage_info["labelTh"],
                    "badgeClass": stage_info["badgeClass"],
                    "confidence": conf_pct,
                    "bbox": f"[{int(x1)}, {int(y1)}, {int(w)}, {int(h)}]",
                    "image": crop_image_url,
                    "frame": frame_count,
                    "progress": progress_pct
                }
                
                yield {
                    "type": "detection",
                    "data": detection_record,
                    "progress": progress_pct
                }
                
            yield {
                "type": "progress",
                "frame": frame_count,
                "total_frames": total_frames,
                "progress": progress_pct,
                "elapsed_sec": round(elapsed_sec, 1)
            }
            
    finally:
        cap.release()
                
    yield {
        "type": "complete",
        "total_detected": banana_counter,
        "processed_frames": frame_count
    }


def analyze_image_full(bgr_image, conf_threshold=0.35, output_crops_dir=None, image_counter_start=1):
    """
    วิเคราะห์ภาพนิ่ง 1 ภาพแบบ Full Pipeline:
    1. ส่งทั้งภาพเข้า Roboflow → ได้ Bounding Box ของกล้วยทุกอัน
    2. Crop ตาม BBox แต่ละอัน → ส่งเข้า MobileNetV2 จำแนก 4 คลาส
    3. วาด annotation + bbox บนภาพ full + แปลง crop แต่ละชิ้นเป็น Base64 Data URL (RAM)
    4. ส่งคืนผลลัพธ์ทุก detection + ภาพ annotated full (base64)
    """
    if output_crops_dir:
        os.makedirs(output_crops_dir, exist_ok=True)

    h_img, w_img = bgr_image.shape[:2]
    predictions = detect_bananas_in_frame(bgr_image)

    # กรอง predictions ตาม conf_threshold (แต่ถ้าไม่มีเลยให้ผ่อนลงมา 0.20)
    filtered = [p for p in predictions if p.get("confidence", 0) >= conf_threshold]
    if not filtered and predictions:
        filtered = [p for p in predictions if p.get("confidence", 0) >= 0.20]

    # สำเนาภาพสำหรับวาด annotation รวม
    annotated_full = bgr_image.copy()

    detections = []
    banana_counter = image_counter_start

    STAGE_COLORS = {
        "Fresh":    (129, 185, 16),
        "fresh":    (129, 185, 16),
        "Unripe":   (166, 184, 20),
        "unripe":   (166, 184, 20),
        "Ripe":     (11, 158, 245),
        "ripe":     (11, 158, 245),
        "Overripe": (12, 88, 234),
        "overripe": (12, 88, 234),
        "Rotten":   (72, 29, 225),
        "rotten":   (72, 29, 225),
    }

    for pred in sorted(filtered, key=lambda p: p.get("confidence", 0), reverse=True):
        bx = float(pred["x"])
        by = float(pred["y"])
        bw = float(pred["width"])
        bh = float(pred["height"])
        bconf = float(pred.get("confidence", 0.0))
        bcls  = str(pred.get("class", "banana"))

        x1 = max(0, int(bx - bw / 2))
        y1 = max(0, int(by - bh / 2))
        x2 = min(w_img, int(bx + bw / 2))
        y2 = min(h_img, int(by + bh / 2))

        cropped = bgr_image[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        # Stage 2: MobileNetV2 classify
        stage_key, stage_conf = classify_crop(cropped)
        stage_info  = STAGES_INFO.get(stage_key, STAGES_INFO["Ripe"])
        conf_pct    = round(stage_conf * 100, 1)
        color_bgr   = STAGE_COLORS.get(stage_key, (200, 200, 200))

        # วาด bbox บนภาพ full
        cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color_bgr, 3)
        label = f"#{banana_counter} {stage_key} {int(conf_pct)}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated_full, (x1, y1 - th - bl - 6), (x1 + tw + 10, y1), color_bgr, -1)
        cv2.putText(annotated_full, label, (x1 + 5, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2, cv2.LINE_AA)

        # แปลง crop ที่มี annotation เป็น Base64 Data URL
        annotated_crop = annotate_crop_image(cropped, stage_info, banana_counter, conf_pct)
        crop_image_url = crop_to_base64(annotated_crop)
        if output_crops_dir:
            ts_ms = int(time.time() * 1000) % 100000
            crop_filename = f"crop_img_{banana_counter:04d}_{ts_ms}.jpg"
            cv2.imwrite(os.path.join(output_crops_dir, crop_filename), annotated_crop)
            crop_image_url = f"/static/crops/{crop_filename}"

        detections.append({
            "id":          f"BN-{banana_counter:04d}",
            "timestamp":   "image",
            "stageKey":    stage_info["key"],
            "stageLabel":  stage_info["labelTh"],
            "badgeClass":  stage_info["badgeClass"],
            "confidence":  conf_pct,
            "roboflow": {
                "detected":    True,
                "class":       bcls,
                "confidence":  round(bconf * 100, 1),
                "box":         [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
            },
            "bbox":        f"[{x1}, {y1}, {int(bw)}, {int(bh)}]",
            "box_coords":  {"x": x1, "y": y1, "w": int(bw), "h": int(bh)},
            "image":       crop_image_url,
        })
        banana_counter += 1

    # ถ้า Roboflow ไม่พบอะไรเลย → ใช้ทั้งภาพเป็น fallback
    if not detections:
        stage_key, stage_conf = classify_crop(bgr_image)
        stage_info  = STAGES_INFO.get(stage_key, STAGES_INFO["Ripe"])
        conf_pct    = round(stage_conf * 100, 1)
        annotated_crop = annotate_crop_image(bgr_image, stage_info, banana_counter, conf_pct)
        crop_image_url = crop_to_base64(annotated_crop)
        if output_crops_dir:
            ts_ms = int(time.time() * 1000) % 100000
            crop_filename = f"crop_img_{banana_counter:04d}_{ts_ms}.jpg"
            cv2.imwrite(os.path.join(output_crops_dir, crop_filename), annotated_crop)
            crop_image_url = f"/static/crops/{crop_filename}"
        detections.append({
            "id":          f"BN-{banana_counter:04d}",
            "timestamp":   "image",
            "stageKey":    stage_info["key"],
            "stageLabel":  stage_info["labelTh"],
            "badgeClass":  stage_info["badgeClass"],
            "confidence":  conf_pct,
            "roboflow": {"detected": False, "class": "none", "confidence": 0.0, "box": None},
            "bbox":        "[0, 0, 0, 0]",
            "box_coords":  {"x": 0, "y": 0, "w": w_img, "h": h_img},
            "image":       crop_image_url,
        })

    # แปลงภาพ annotated full เป็น base64 JPEG
    _, buf = cv2.imencode(".jpg", annotated_full, [cv2.IMWRITE_JPEG_QUALITY, 88])
    annotated_b64 = "data:image/jpeg;base64," + __import__("base64").b64encode(buf.tobytes()).decode()

    return {
        "detections":    detections,
        "total":         len(detections),
        "annotated_image": annotated_b64,
        "image_size":    {"w": w_img, "h": h_img},
    }


# โหลด Client และโมเดล MobileNetV2 เมื่อเริ่มต้น
init_roboflow_client()
init_classifier_model()
