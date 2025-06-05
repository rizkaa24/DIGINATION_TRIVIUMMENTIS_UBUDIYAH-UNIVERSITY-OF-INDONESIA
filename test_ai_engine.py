from app.ai_engine import HerbalAI

def test_recommendation():
    ai = HerbalAI("data/tanaman_herbal.csv")
    hasil = ai.recommend_by_symptom("batuk dan demam")
    assert len(hasil) > 0
