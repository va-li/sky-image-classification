#!./venv/bin/python3

import io
import torch
from torchvision.transforms import v2
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from classifier.model import MultiLabelClassificationMobileNetV3Large
import time
from pathlib import Path
from typing import List, Dict, Any

torch.backends.cudnn.deterministic = True

# Initialize model and load pretrained weights
model = MultiLabelClassificationMobileNetV3Large(num_classes=5)
try:
    model.load_state_dict(
        torch.load("./classifier/best_model.pth", map_location=torch.device("cpu"))
    )
except FileNotFoundError:
    raise Exception("No pretrained weights found")

model.eval()

# Image preprocessing
transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# dataset
dataset_path = Path("./dataset")
CLASS_LABELS = (dataset_path / "synsets.txt").read_text().split("\n")[:-1] # Remove last empty line

def parse_image_path_and_labels(label_file_path: Path | str, class_labels: list) -> Dict[str, List[Dict[str, Any]]]:
    """ Parse image path and labels from a line in the dataset label file (default.txt)"""
    
    ground_truth_raw = label_file_path.read_text().split("\n")
    ground_truth = {}
    for line in ground_truth_raw:
        if not line:
            # Skip empty lines
            continue
        # Split line by whitespace
        parts = line.split()
        
        if len(parts) < 2:
            # Skip if there are no labels
            continue
        
        # First part is the image path
        image_path = Path(parts[0]).name
        # The rest are labels
        labels_num_sparse = [ int(label_num) for label_num in parts[1:] ]
        # labels_text_sparse = [ class_labels[label_num] for label_num in labels_num_sparse ]
        
        labels_all_binary = [ {"label": class_labels[i], "ground_truth": 1 } if i in labels_num_sparse else {"label": class_labels[i], "ground_truth": 0 } for i in range(len(class_labels)) ]
        
        ground_truth[image_path] = labels_all_binary

    return ground_truth

GROUND_TRUTH = parse_image_path_and_labels(dataset_path / "default.txt", CLASS_LABELS)

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
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            start = time.time()
            predictions = model(input_tensor).squeeze()
            end = time.time()

        # Prepare response
        labels = [
            {"label": label, "prediction_score": float(score)}
            for label, score in zip(CLASS_LABELS, predictions)
        ]

        inference_time_ms = (end - start) * 1000
        
        # pull ground truth labels if available
        file_name = Path(file.filename).name
        if file_name in GROUND_TRUTH:
            ground_truth = GROUND_TRUTH[file_name]
            # merge ground truth and predictions into one list
            labels = [ {**pred, **gt} for pred, gt in zip(labels, ground_truth) ]
        else:
            ground_truth = None
        
        responseContent = {
            "labels": labels,
            "inference_time_ms": inference_time_ms,
        }

        return JSONResponse(content=responseContent)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
async def index():
    return FileResponse("public/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
