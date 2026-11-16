import os
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# โหลดค่าตัวแปรจาก .env
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "banana-detection-ieo3p/2")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/MobileNetV2_banana_model.keras")
DEFAULT_FRAME_INTERVAL = int(os.getenv("DEFAULT_FRAME_INTERVAL", 10))
DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF_THRESHOLD", 0.75))

# รายละเอียด Stage และการจัดหมวดหมู่สี (4 คลาส)
STAGES_INFO = {
    "Unripe": {
        "key": "Unripe",
        "labelTh": "ดิบ (Unripe)",
        "badgeClass": "bg-emerald-100 text-emerald-800 border-emerald-300",
        "hexColor": "#10b981",
        "bgrColor": (129, 185, 16)
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
    },
    "Rotten": {
        "key": "Rotten",
        "labelTh": "เน่าเสีย (Rotten)",
        "badgeClass": "bg-rose-100 text-rose-800 border-rose-300",
        "hexColor": "#e11d48",
        "bgrColor": (72, 29, 225)
    }
}

# ลำดับ 4 คลาสตามโมเดล MobileNetV2 ที่เทรนไว้ (Class A -> D)
CLASSES_RIPENESS = ["Unripe", "Ripe", "Overripe", "Rotten"]

# Global Objects
roboflow_client = None
classifier_model = None
classifier_type = None  # 'keras' or 'torch'


def init_roboflow_client():
    """ตั้งค่า Roboflow Client สำหรับ YOLOv8 Detection"""
    global roboflow_client
    api_key = os.getenv("ROBOFLOW_API_KEY", "")
    if not api_key or "your_" in api_key or "2He***" in api_key:
        print("[Detector] Roboflow API Key ยังไม่ได้ตั้งค่าหรือเป็นค่าตัวอย่างใน .env")
        roboflow_client = None
        return False
    try:
        from inference_sdk import InferenceHTTPClient
        roboflow_client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=api_key
        )
        print(f"[Detector] เชื่อมต่อ Roboflow Client สำเร็จ (Model: {ROBOFLOW_MODEL_ID})")
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
    (Unripe, Ripe, Overripe, Rotten) พร้อมค่า Confidence
    """
    global classifier_model, classifier_type
    
    if classifier_model is None:
        raise RuntimeError("❌ [Detector] โมเดล MobileNetV2 ยังไม่ได้ถูกโหลดเข้าหน่วยความจำ!")

    if bgr_crop is None or bgr_crop.size == 0:
        raise ValueError("❌ [Detector] ภาพ Crop ว่างเปล่า ไม่สามารถส่งเข้าโมเดลได้")

    # โมเดล TensorFlow/Keras
    if classifier_type == "keras":
        import tensorflow as tf
        # แปลง BGR -> RGB และปรับขนาดเป็น 224x224 ตาม Input Shape ของ MobileNetV2
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        # Preprocessing เฉพาะของ MobileNetV2 (scale pixel สู่ช่วง [-1, 1])
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        
        preds = classifier_model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        
        # แมป Class ID สู่ 4 Stages:
        # Index 0: Class A (Unripe), 1: Class B (Ripe), 2: Class C (Overripe), 3: Class D (Rotten)
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
    ใช้เฉพาะเมื่อไม่ได้ใส่ Roboflow API Key สำหรับ Stage 1
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 40, 50])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions = []
    min_area = (frame.shape[0] * frame.shape[1]) * 0.008
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            if aspect_ratio >= 1.2:
                predictions.append({
                    "x": x + w / 2,
                    "y": y + h / 2,
                    "width": w,
                    "height": h,
                    "confidence": min(0.94, 0.78 + (area / (frame.shape[0] * frame.shape[1])) * 2),
                    "class": "banana"
                })
    return predictions


def detect_bananas_in_frame(frame, temp_image_path="temp_frame.jpg"):
    """
    เรียกใช้ Roboflow YOLOv8 เพื่อหาตำแหน่งกล้วยในภาพ
    """
    global roboflow_client
    if roboflow_client is not None:
        try:
            cv2.imwrite(temp_image_path, frame)
            res = roboflow_client.infer(temp_image_path, model_id=ROBOFLOW_MODEL_ID)
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


def process_video_generator(video_path, frame_interval=DEFAULT_FRAME_INTERVAL, conf_threshold=DEFAULT_CONF_THRESHOLD, output_crops_dir="static/crops"):
    """
    ประมวลผลวิดีโอแบบ Generator ส่งผลลัพธ์ผ่าน MobileNetV2 100%
    """
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
    temp_frame_path = os.path.join(output_crops_dir, "_temp_frame.jpg")
    
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
            
            # Stage 1: ตรวจจับตำแหน่งด้วย YOLOv8
            predictions = detect_bananas_in_frame(frame, temp_frame_path)
            
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
                crop_filename = f"crop_bn_{banana_counter:04d}_{frame_count}.jpg"
                crop_path = os.path.join(output_crops_dir, crop_filename)
                cv2.imwrite(crop_path, annotated_crop)
                
                detection_record = {
                    "id": f"BN-{banana_counter:04d}",
                    "timestamp": f"{elapsed_sec:.1f}s",
                    "stageKey": stage_info["key"],
                    "stageLabel": stage_info["labelTh"],
                    "badgeClass": stage_info["badgeClass"],
                    "confidence": conf_pct,
                    "bbox": f"[{int(x1)}, {int(y1)}, {int(w)}, {int(h)}]",
                    "image": f"/static/crops/{crop_filename}",
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
        if os.path.exists(temp_frame_path):
            try:
                os.remove(temp_frame_path)
            except Exception:
                pass
                
    yield {
        "type": "complete",
        "total_detected": banana_counter,
        "processed_frames": frame_count
    }


# โหลด Client และโมเดล MobileNetV2
init_roboflow_client()
init_classifier_model()
