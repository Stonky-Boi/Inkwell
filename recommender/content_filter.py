import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_book_features(books_dataframe: pd.DataFrame) -> pd.DataFrame:
    if books_dataframe.empty:
        raise ValueError("The provided books dataframe is empty.")
        
    required_columns = ["ISBN", "Book-Title", "Book-Author", "Publisher"]
    for column_name in required_columns:
        if column_name not in books_dataframe.columns:
            raise KeyError(f"Missing required column: {column_name}")

    books_dataframe_filled = books_dataframe.fillna("")

    combined_features = (
        books_dataframe_filled["Book-Title"].astype(str) + " " +
        books_dataframe_filled["Book-Author"].astype(str) + " " +
        books_dataframe_filled["Publisher"].astype(str)
    )
    
    books_dataframe_filled["combined_features"] = combined_features
    
    return books_dataframe_filled

def build_tfidf_feature_matrix(books_dataframe_with_features: pd.DataFrame):
    if "combined_features" not in books_dataframe_with_features.columns:
        raise KeyError("The combined_features column is missing from the dataframe.")

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(books_dataframe_with_features["combined_features"])
    
    return tfidf_matrix

def get_similarity_score(element: tuple) -> float:
    return element[1]

def get_content_scores(target_isbn: str, tfidf_matrix, books_dataframe: pd.DataFrame, pool_size: int = 50) -> dict:
    if target_isbn not in books_dataframe["ISBN"].values:
        raise ValueError(f"The ISBN {target_isbn} is not present in the books database.")

    book_index_series = books_dataframe.index[books_dataframe["ISBN"] == target_isbn]
    if book_index_series.empty:
         raise ValueError("Could not locate the index for the provided ISBN.")
         
    book_index_position = book_index_series[0]

    target_vector = tfidf_matrix[book_index_position]
    similarity_scores_array = cosine_similarity(target_vector, tfidf_matrix).flatten()
    
    similarity_scores_list = list(enumerate(similarity_scores_array))
    sorted_scores = sorted(similarity_scores_list, key=get_similarity_score, reverse=True)
    
    content_scores_dictionary = {}
    
    for index_value, score in sorted_scores[1:pool_size + 1]:
        recommended_isbn = books_dataframe.iloc[index_value]["ISBN"]
        content_scores_dictionary[recommended_isbn] = float(score)
        
    return content_scores_dictionary