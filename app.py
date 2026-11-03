# USING 91% ACCURACY MODEL 
import os
import pickle
import numpy as np
import joblib

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------
# 1. File paths
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fake_news_lstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "fake_news_tokenizer.pkl")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


# --------------------------------
# 2. Load trained model and tokenizer
# --------------------------------

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)


# These values must match your training code
MAX_LEN = 500


# --------------------------------
# 3. Create FastAPI application
# --------------------------------

app = FastAPI(title="Fake News Detection System")

templates = Jinja2Templates(directory=TEMPLATE_DIR)


# --------------------------------
# 4. Input data format
# --------------------------------

class NewsInput(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


# --------------------------------
# 5. Home page
# --------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# --------------------------------
# 6. Prediction page
# --------------------------------

@app.get("/predict", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={}
    )


# --------------------------------
# 7. Prediction API
# --------------------------------

@app.post("/predict")
async def predict_news(news: NewsInput):

    try:
        title = news.title.strip()
        article = news.text.strip()

        if not title or not article:
            raise HTTPException(
                status_code=400,
                detail="Please enter both headline and article content."
            )

        # Combine title and article exactly as during training
        content = title + " " + article

        # Convert text into sequences
        sequence = tokenizer.texts_to_sequences([content])

        # Pad sequence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # Model prediction
        prediction = model.predict(padded_sequence, verbose=0)

        probability = float(prediction[0][0])

        # Label mapping: 0 = Real, 1 = Fake
        if probability >= 0.5:
            result = "Fake News"
            confidence = probability * 100
        else:
            result = "Real News"
            confidence = (1 - probability) * 100

        return {
            "prediction": result,
            "confidence": round(confidence, 2),
            "fake_probability": round(probability * 100, 2),
            "real_probability": round((1 - probability) * 100, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )















# # THIS IS USING LINEAR SVM MODEL WITH CALIBRATED PROBABILITIES 

# import os
# import joblib

# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel, Field


# # --------------------------------
# # 1. File paths
# # --------------------------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(
#     BASE_DIR, "calibrated_fake_news_svm.joblib"
# )
# TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


# # --------------------------------
# # 2. Load calibrated SVM model
# # --------------------------------

# model = joblib.load(MODEL_PATH)

# print("Calibrated SVM model loaded successfully!")


# # --------------------------------
# # 3. Create FastAPI application
# # --------------------------------

# app = FastAPI(title="Fake News Detection System")

# templates = Jinja2Templates(directory=TEMPLATE_DIR)


# # --------------------------------
# # 4. Input data format
# # --------------------------------

# class NewsInput(BaseModel):
#     title: str = Field(..., min_length=1)
#     text: str = Field(..., min_length=1)


# # --------------------------------
# # 5. Home page
# # --------------------------------

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):

#     return templates.TemplateResponse(
#         request=request,
#         name="index.html",
#         context={}
#     )


# # --------------------------------
# # 6. Prediction page
# # --------------------------------

# @app.get("/predict", response_class=HTMLResponse)
# async def prediction_page(request: Request):

#     return templates.TemplateResponse(
#         request=request,
#         name="predict.html",
#         context={}
#     )


# # --------------------------------
# # 7. Prediction API
# # --------------------------------


# @app.post("/predict")
# async def predict_news(news: NewsInput):

#     try:
#         title = news.title.strip()
#         article = news.text.strip()

#         if not title or not article:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Please enter both headline and article content."
#             )

#         # Combine headline and article
#         content = title + " " + article

#         # Get calibrated probabilities
#         probabilities = model.predict_proba([content])[0]

#         # Get class positions
#         classes = list(model.classes_)

#         real_index = classes.index(0)
#         fake_index = classes.index(1)

#         real_probability = float(probabilities[real_index])
#         fake_probability = float(probabilities[fake_index])

#         # Determine prediction
#         if fake_probability >= 0.5:
#             result = "Fake News"
#             confidence = fake_probability * 100
#         else:
#             result = "Real News"
#             confidence = real_probability * 100

#         return {
#             "prediction": result,
#             "confidence": round(confidence, 2),
#             "fake_probability": round(fake_probability * 100, 2),
#             "real_probability": round(real_probability * 100, 2)
#         }

#     except HTTPException:
#         raise

#     except Exception as e:
#         import traceback

#         traceback.print_exc()

#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction error: {str(e)}"
#         )
