from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
from typing import Optional
import secrets
from utils import convert_bdf_to_tensor_list

app = FastAPI()
security = HTTPBasic()

# Модель для пользователя
class User(BaseModel):
    username: str
    password: str

# Фейковая база данных пользователей
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "admin",
        "hashed_password": secrets.token_hex(16)
    }
}

# Токены сессий
active_sessions = set()

# Загрузка ONNX модели при старте приложения
onnx_model_path = "model.onnx"
try:
    ort_session = ort.InferenceSession(onnx_model_path)
except Exception as e:
    print(f"Failed to load ONNX model: {e}")
    ort_session = None

# Эндпоинты

@app.post("/login/")
async def login(user: User):
    if user.username not in fake_users_db or user.password != fake_users_db[user.username]["password"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    session_token = secrets.token_urlsafe(32)
    active_sessions.add(session_token)
    return {"access_token": session_token, "token_type": "bearer"}

@app.post("/logout/")
async def logout(token: str):
    if token in active_sessions:
        active_sessions.remove(token)
        return {"message": "Successfully logged out"}
    raise HTTPException(status_code=400, detail="Invalid token")

def verify_token(token: str):
    if token not in active_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True

@app.post("/getModelResult/")
async def get_model_result(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    if not ort_session:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.filename.endswith('.bdf'):
        raise HTTPException(status_code=400, detail="Only .bdf files are accepted")
    
    try:
        # Сохраняем временный файл
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        
        input_data = convert_bdf_to_tensor_list(file_path)[0]
        
        # Получаем имена входных узлов модели
        input_name = ort_session.get_inputs()[0].name
        
        # Выполняем анализ
        outputs = ort_session.run(None, {input_name: input_data})
        
        # Удаляем временный файл
        os.remove(file_path)
        
        # Преобразуем numpy array в список для JSON
        result = outputs[0].tolist()
        
        return {"result": result, "status": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
