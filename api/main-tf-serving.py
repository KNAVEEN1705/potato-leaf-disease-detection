from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()
# TensorFlow Serving endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am GOAT"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = image.convert('RGB')  # Ensure image is in RGB mode
        image = image.resize((256, 256))  # Resize to the expected input size
        image = np.array(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

def predict(image: np.ndarray):
    img_batch = np.expand_dims(image, axis=0)  # Create a batch
    json_data = {"instances": img_batch.tolist()}  # Convert to list for JSON

    try:
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()  # Raise an error for bad responses
        predictions = response.json()["predictions"][0]
        return predictions
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        predictions = predict(image)

        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)

        return {"class": predicted_class, "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
