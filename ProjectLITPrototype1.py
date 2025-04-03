import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    def __init__(self):
        self.books = []
        self.authors = []
        self.years = []
        self.texts = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
    
    def add_books(self, book_data):
        for book in book_data:
            combined_text = f"{book['title']} {book['author']} {book['year']} {book['text']}"
            self.books.append(book['title'])
            self.authors.append(book['author'])
            self.years.append(book['year'])
            self.texts.append(combined_text)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
    
    def recommend(self, user_books):
        if not self.books:
            print("База данных книг пуста. Добавьте книги перед рекомендацией.")
            return []
        user_texts = [f"{book['title']} {book['author']} {book['year']} {book['text']}" for book in user_books]
        user_tfidf = self.vectorizer.transform(user_texts)
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
        recommendations = []
        for sim in similarities:
            max_index = np.argmax(sim)
            recommended_book = (self.books[max_index], sim[max_index])
            recommendations.append(recommended_book)
        return recommendations

if __name__ == "__main__":
    base_books = [
        {"title": "Книга о Python", "author": "Иван Иванов", "year": "2020", "text": "Это текст книги про программирование."},
        {"title": "Приключенческий роман", "author": "Петр Петров", "year": "2015", "text": "История о приключениях героя."},
        {"title": "Философия жизни", "author": "Сергей Сергеев", "year": "2010", "text": "Философские размышления о жизни."}
    ]
    recommender = BookRecommender()
    recommender.add_books(base_books)
    user_books = [{"title": "Программирование для начинающих", "author": "Алексей Смирнов", "year": "2022", "text": "Как писать код и программировать?"}]
    recommendations = recommender.recommend(user_books)
    print("Рекомендованные книги:")
    for i, (title, score) in enumerate(recommendations):
        print(f"Для запроса '{user_books[i]['title']}':")
        print(f"  - {title} (сходство: {score:.2f})")
