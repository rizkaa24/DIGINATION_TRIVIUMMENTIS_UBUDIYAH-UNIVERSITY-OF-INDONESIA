from fastapi import FastAPI
from app.ai_engine import HerbalAI

app = FastAPI()
model = HerbalAI("data/tanaman_herbal.csv")

@app.get("/rekomendasi")
def get_rekomendasi(gejala: str):
    hasil = model.recommend_by_symptom(gejala)
    return {"rekomendasi": hasil}
