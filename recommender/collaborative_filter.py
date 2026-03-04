import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

def filter_and_merge_data(ratings_dataframe: pd.DataFrame, users_dataframe: pd.DataFrame, minimum_ratings: int = 50) -> pd.DataFrame:
    if ratings_dataframe.empty or users_dataframe.empty:
        raise ValueError("Dataframes cannot be empty.")

    # Integrate demographic data to ensure rating quality
    valid_users = users_dataframe[users_dataframe['Age'].between(5, 100, inclusive='both')]
    merged_data = pd.merge(ratings_dataframe, valid_users, on='User-ID', how='inner')

    user_counts = merged_data['User-ID'].value_counts()
    active_users = user_counts[user_counts >= minimum_ratings].index
    
    book_counts = merged_data['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= minimum_ratings].index

    filtered_data = merged_data[merged_data['User-ID'].isin(active_users)]
    final_filtered_data = filtered_data[filtered_data['ISBN'].isin(popular_books)]

    return final_filtered_data

def create_user_item_matrix(filtered_data: pd.DataFrame) -> pd.DataFrame:
    user_item_matrix = filtered_data.pivot_table(
        index='User-ID',
        columns='ISBN',
        values='Book-Rating'
    ).fillna(0.0)
    
    return user_item_matrix

def train_svd_model(user_item_matrix: pd.DataFrame, latent_factors: int = 20) -> tuple:
    if user_item_matrix.empty:
        raise ValueError("User-item matrix is empty.")

    matrix_values = user_item_matrix.values
    user_ratings_mean = np.mean(matrix_values, axis=1)
    normalized_matrix = matrix_values - user_ratings_mean.reshape(-1, 1)

    # Apply SVD
    u_matrix, sigma_array, vt_matrix = svds(normalized_matrix, k=latent_factors)

    # Explicitly catch None to satisfy Pylance and prevent silent downstream failures
    if u_matrix is None or sigma_array is None or vt_matrix is None:
        raise RuntimeError("Singular Value Decomposition failed to return valid matrices.")

    sigma_diagonal = np.diag(sigma_array)

    # Reconstruct the matrix to get predictions
    predicted_ratings = np.dot(np.dot(u_matrix, sigma_diagonal), vt_matrix) + user_ratings_mean.reshape(-1, 1)
    
    # Calculate Evaluation Metrics
    actual_non_zero = matrix_values[matrix_values.nonzero()].flatten()
    predicted_non_zero = predicted_ratings[matrix_values.nonzero()].flatten()
    
    rmse_score = np.sqrt(mean_squared_error(actual_non_zero, predicted_non_zero))
    mae_score = mean_absolute_error(actual_non_zero, predicted_non_zero)
    
    metrics_dictionary = {
        "RMSE": rmse_score,
        "MAE": mae_score
    }

    # vt_matrix transpose gives us the item embeddings
    item_embeddings = vt_matrix.T
    item_similarity_matrix = cosine_similarity(item_embeddings)
    
    # Create a dataframe for easy ISBN lookup
    item_similarity_dataframe = pd.DataFrame(
        item_similarity_matrix, 
        index=user_item_matrix.columns, 
        columns=user_item_matrix.columns
    )

    return item_similarity_dataframe, metrics_dictionary

def get_similarity_score(element: tuple) -> float:
    return element[1]

def get_collaborative_scores(item_similarity_dataframe: pd.DataFrame, target_isbn: str, pool_size: int = 50) -> dict:
    if target_isbn not in item_similarity_dataframe.index:
        raise KeyError(f"The ISBN {target_isbn} was not found in the collaborative matrix.")

    similarity_series = item_similarity_dataframe.loc[target_isbn]
    similarity_scores_list = list(enumerate(similarity_series.values))
    
    sorted_scores = sorted(similarity_scores_list, key=get_similarity_score, reverse=True)
    
    collaborative_scores_dictionary = {}
    
    # Start at 1 to skip the target book itself
    for index_value, score in sorted_scores[1:pool_size + 1]:
        recommended_isbn = item_similarity_dataframe.columns[index_value]
        collaborative_scores_dictionary[recommended_isbn] = float(score)

    return collaborative_scores_dictionary