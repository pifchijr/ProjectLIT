import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

books = [
    "Над пропастью во ржи — это история о молодом человеке, разочаровавшемся в окружающем мире.",
    "Убить пересмешника поднимает серьезные темы, такие как расовое неравенство и нравственное взросление.",
    "1984 — антиутопический роман о тоталитарном обществе, где правительство контролирует все.",
    "Гордость и предубеждение исследует темы любви, брака и социального класса в Англии XIX века.",
    "Моби Дик рассказывает о навязчивом стремлении моряка убить гигантского белого кита."
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books)

def recommend_books(book_idx, tfidf_matrix, books, top_n=3):
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix)
    similar_books_idx = cosine_sim.argsort()[0][-top_n-1:-1][::-1]
    return [books[i] for i in similar_books_idx]

recommended_books = recommend_books(0, tfidf_matrix, books)
for idx, book in enumerate(recommended_books, 1):
    print(f"{idx}. {book}")
