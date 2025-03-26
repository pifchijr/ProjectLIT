import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Пример базы данных книг с описаниями
books = [
    "Над пропастью во ржи — это история о молодом человеке, разочаровавшемся в окружающем мире.",
    "Убить пересмешника поднимает серьезные темы, такие как расовое неравенство и нравственное взросление.",
    "1984 — антиутопический роман о тоталитарном обществе, где правительство контролирует все.",
    "Гордость и предубеждение исследует темы любви, брака и социального класса в Англии XIX века.",
    "Моби Дик рассказывает о навязчивом стремлении моряка убить гигантского белого кита."
]

# Преобразуем текст в TF-IDF векторы
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books)

# Функция для получения рекомендаций
def recommend_books(book_idx, tfidf_matrix, books, top_n=3):
    # Вычисление сходства между книгами
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix)
    # Получаем индексы наиболее похожих книг
    similar_books_idx = cosine_sim.argsort()[0][-top_n-1:-1][::-1]
    # Возвращаем наиболее похожие книги
    return [books[i] for i in similar_books_idx]

# Пример: Рекомендации для книги с индексом 0 (Над пропастью во ржи)
recommended_books = recommend_books(0, tfidf_matrix, books)
for idx, book in enumerate(recommended_books, 1):
    print(f"{idx}. {book}")
