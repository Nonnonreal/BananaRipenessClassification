import cv2
import torch
from torchvision import transforms
from PIL import Image
from inference_sdk import InferenceHTTPClient

# 1. ตั้งค่า Roboflow Client สำหรับ YOLOv8 Detection
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="2He*****************"
)

# 2. โหลดโมเดล MobileNetV2 สำหรับจำแนกความสุก (สมมติว่าเทรนและเซฟน้ำหนักไว้แล้วเป็น ripeness_model.pth)
ripeness_model = torch.load("ripeness_model.pth", map_location="cpu")
ripeness_model.eval()

# Transform สำหรับแปลงภาพให้ MobileNetV2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# รายชื่อคลาสความสุก (4 คลาส)
CLASSES_RIPENESS = ["Unripe (ดิบ)", "Ripe (สุก)", "Overripe (งอม)", "Rotten (เน่าเสีย)"]

def process_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # --- เทคนิคจำกัดเฟรม (Frame Skipping) ---
        # ประมวลผลเฉพาะเฟรมที่หารด้วย frame_interval ลงตัวเท่านั้น (เช่น ทุกๆ 10 เฟรม)
        if frame_count % frame_interval != 0:
            continue
            
        # บันทึกเฟรมชั่วคราวเพื่อส่งให้ Roboflow API
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)
        
        # 3. Stage 1: ให้ YOLOv8 ตรวจจับกล้วยในภาพ
        result = CLIENT.infer(temp_image_path, model_id="banana-detection-ieo3p/2")
        
        # วนลูปตาม Bounding Box ที่ YOLOv8 หาเจอ
        for prediction in result.get("predictions", []):
            x = prediction["x"]
            y = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            
            # คำนวณพิกัดเพื่อ Crop ภาพกล้วยแต่ละลูก
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Crop ภาพจากเฟรมวิดีโอต้นฉบับ
            cropped_banana = frame[y1:y2, x1:x2]
            if cropped_banana.size == 0:
                continue
                
            # แปลงภาพ OpenCV เป็น PIL Image เพื่อเข้า MobileNetV2
            pil_img = Image.fromarray(cv2.cvtColor(cropped_banana, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_img).unsqueeze(0)
            
            # 4. Stage 2: ให้ MobileNetV2 ทำนายความสุก
            with torch.no_grad():
                outputs = ripeness_model(input_tensor)
                _, predicted_class = torch.max(outputs, 1)
                ripeness_label = CLASSES_RIPENESS[predicted_class.item()]
                
            print(f"Frame {frame_count}: เจอวัตถุ -> ความสุก: {ripeness_label}")
            
    cap.release()

# ทดลองเรียกใช้งานฟังก์ชัน (เช่น ข้ามทุกๆ 15 เฟรม)
# process_video("path_to_video.mp4", frame_interval=15)