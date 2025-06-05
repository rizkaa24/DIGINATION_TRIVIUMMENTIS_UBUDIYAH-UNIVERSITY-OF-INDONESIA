import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class HerbalAI:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer()
        self.nn = NearestNeighbors(n_neighbors=3, metric='cosine')
        self._train_model()

    def _train_model(self):
        tfidf_matrix = self.vectorizer.fit_transform(self.df['kandungan'].fillna(''))
        self.nn.fit(tfidf_matrix)

    def recommend_by_symptom(self, symptom_desc):
        input_vec = self.vectorizer.transform([symptom_desc])
        distances, indices = self.nn.kneighbors(input_vec)
        return self.df.iloc[indices[0]].to_dict(orient='records')
