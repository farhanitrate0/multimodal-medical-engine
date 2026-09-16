import base64, io, cv2, torch, json
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
from PIL import Image
from torchvision import transforms
from model import get_medical_model
from gradcam import GradCAMEngine

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_medical_model().to(device)
try:
    model.load_state_dict(torch.load("fine_tuned_medical_resnet.pth", map_location=device, weights_only=True))
    print("Loaded fine-tuned medical weights.")
except Exception:
    print("Using baseline initialized weights.")
model.eval()

grad_cam = GradCAMEngine(model, model.layer4[-1])

mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands_tracker = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_pipeline(payload_str: str):
    data = json.loads(payload_str)
    b64_data = data["image"]
    mode = data["mode"]
    
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    
    img_bytes = base64.b64decode(b64_data)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB")), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))
    
    conf = 0.0
    status_msg = ""
    overlay = img_resized.copy()

    if mode == 'medical':
        tensor = transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad(): 
            conf = torch.sigmoid(model(tensor)).item()
        
        heatmap = grad_cam.generate_heatmap(tensor)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(heatmap_color, 0.4, img_resized, 0.6, 0)
        status_msg = f"Pathology Classification Confidence: {conf*100:.1f}%"

    elif mode == 'spatial':
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = mp_face.process(rgb_img)
        hand_results = hands_tracker.process(rgb_img)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for lm in face_landmarks.landmark[::5]:
                    cx, cy = int(lm.x * 224), int(lm.y * 224)
                    cv2.circle(overlay, (cx, cy), 1, (0, 255, 0), -1)
                    
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark[::3]:
                    cx, cy = int(lm.x * 224), int(lm.y * 224)
                    cv2.circle(overlay, (cx, cy), 2, (255, 165, 0), -1)
                    
        status_msg = "Spatial Tracking Active (Face & Hands)"

    _, buffer = cv2.imencode('.jpg', overlay)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return conf, status_msg, encoded_img

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            conf, status_msg, overlay = await asyncio.to_thread(process_pipeline, raw_data)
            await websocket.send_json({
                "confidence": conf,
                "status": status_msg,
                "overlay": f"data:image/jpeg;base64,{overlay}"
            })
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")