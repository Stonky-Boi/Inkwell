import pandas as pd
import os
# Include the following if required
# os.chdir("Your directory")

# Load datasets
books_dataframe = pd.read_csv('books.csv')
ratings_dataframe = pd.read_csv('ratings.csv')
book_tags_dataframe = pd.read_csv('book_tags.csv')
to_read_dataframe = pd.read_csv('to_read.csv')
tags_dataframe = pd.read_csv('tags.csv')

# Data Preprocessing
# books_dataframe.isnull().sum()
# print(books_dataframe['language_code'].unique())

# Drop unnecessary columns to reduce data complexity
books_dataframe.drop(columns=['best_book_id','work_id','isbn', 'isbn13','work_ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5'], inplace=True)
# Handle missing values in 'language_code'
books_dataframe['language_code'].fillna('eng', inplace=True)
books_dataframe['language_code'].replace(['en-US', 'en-CA', 'en-GB'], 'eng', inplace=True)
# Fill missing 'original_title' with 'title'
books_dataframe['original_title'].fillna(books_dataframe['title'], inplace=True)
# Fill missing 'original_publication_year' with mean value
books_dataframe['original_publication_year'].fillna(books_dataframe['original_publication_year'].mean(), inplace=True)
# Merge datasets to create user-book interaction data
books_ratings_dataframe = pd.merge(books_dataframe, ratings_dataframe, on='book_id')

# Create User-Item matrix (users as rows, books as columns)
user_book_ratings_matrix = books_ratings_dataframe.pivot_table(index='user_id', columns='book_id', values='rating')
# Fill NaN values with 0 (assumption: missing ratings_dataframe mean unrated)
user_book_ratings_matrix = user_book_ratings_matrix.fillna(0)

# Collaborative Filtering using SVD
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# Convert the user-item matrix to a numpy array
user_item_matrix_array = user_book_ratings_matrix.values

# Apply Singular Value Decomposition (SVD)
number_of_components = 50  # Number of latent factors
svd_model = TruncatedSVD(n_components=number_of_components, random_state=42)
svd_transformed_matrix = svd_model.fit_transform(user_item_matrix_array)

def collaborative_filtering_recommendations(user_id, n=10):
    user_index = user_book_ratings_matrix.index.get_loc(user_id)
    user_ratings = svd_transformed_matrix[user_index, :]
    book_scores = svd_model.components_.T.dot(user_ratings)
    top_books = books_dataframe.iloc[book_scores.argsort()[::-1]]
    return top_books['title'].head(n)

# Model evaluation using RMSE
predicted_matrix = svd_model.inverse_transform(svd_transformed_matrix)
mse = mean_squared_error(user_item_matrix_array[user_item_matrix_array.nonzero()], 
                         predicted_matrix[user_item_matrix_array.nonzero()])
rmse = sqrt(mse)
print(f"RMSE: {rmse}")

# Content-Based Filtering using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a new feature combining title, authors, and language for similarity measurement
books_dataframe['content'] = books_dataframe['title'] + ' ' + books_dataframe['authors'] + ' ' + books_dataframe['language_code']

# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_feature_matrix = tfidf_vectorizer.fit_transform(books_dataframe['content'])
# Compute cosine similarity between books_dataframe
cosine_similarity_matrix = cosine_similarity(tfidf_feature_matrix, tfidf_feature_matrix)

def content_based_recommendations(title, cosine_similarity_matrix=cosine_similarity_matrix, n=10):
    idx = books_dataframe[books_dataframe['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [index[0] for index in sim_scores]
    return books_dataframe['title'].iloc[book_indices]

# Hybrid Approach
def hybrid_recommendations(user_id, title, n=10):
    content_recommendations = content_based_recommendations(title, n=n)
    collaborative_recommendations = collaborative_filtering_recommendations(user_id, n=n)
    hybrid_recommendations_list = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates()
    return hybrid_recommendations_list.head(n)

# Model Evaluation Metrics
from sklearn.metrics import precision_score, recall_score

def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(k)

def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / float(len(actual_set))
 
# Deploy using Streamlit
import streamlit as st

st.title("Book Recommender System")

# Get user input
user_id = st.number_input("Enter User ID", min_value=1)
book_title = st.selectbox("Select a book", books_dataframe['title'].unique())

# Generate recommended books
if st.button("Get Recommendations"):
    recommended_books = hybrid_recommendations(user_id, book_title)
    st.write("Recommended Books:")
    for index, recommended_book in enumerate(recommended_books, 1):
        st.write(f"{index}. {recommended_book}")
