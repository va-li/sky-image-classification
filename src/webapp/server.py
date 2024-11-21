import io
import torch
from torchvision.transforms import v2
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from classifier.models import MultiLabelClassificationMobileNetV3Large
import time

# Initialize model and load pretrained weights
model = MultiLabelClassificationMobileNetV3Large(num_classes=5)
try:
    model.load_state_dict(torch.load('./classifier/best_model.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    print("No pretrained weights found. Model will use random initialization.")

model.eval()

# Image preprocessing
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels (replace with your actual labels)
CLASS_LABELS = ['clouds', 'rain', 'dew', 'clear sky', 'soiling']

app = FastAPI()

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/public", StaticFiles(directory="public", html=True), name="public")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        input_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            start = time.time()
            predictions = model(input_tensor).squeeze()
            end = time.time()
        
        # Prepare response
        results = [
            {"label": label, "score": float(score)}
            for label, score in zip(CLASS_LABELS, predictions)
        ]
        
        inference_time_ms = (end - start) * 1000
        
        return JSONResponse(content={"predictions": results, "inference_time_ms": inference_time_ms})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/')
async def index():
    return FileResponse('public/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
