from fastapi import FastAPI, File, UploadFile
from detector import process_image

app = FastAPI()

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()
    result = process_image(contents)
    return result
