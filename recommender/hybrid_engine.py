import pandas as pd
from recommender.collaborative_filter import get_collaborative_scores
from recommender.content_filter import get_content_scores

def normalize_scores(scores_dictionary: dict) -> dict:
    if not scores_dictionary:
        return {}
        
    maximum_score = max(scores_dictionary.values())
    minimum_score = min(scores_dictionary.values())
    
    if maximum_score == minimum_score:
        return {isbn_key: 1.0 for isbn_key in scores_dictionary.keys()}
        
    normalized_dictionary = {}
    for isbn_key, raw_score in scores_dictionary.items():
        normalized_score = (raw_score - minimum_score) / (maximum_score - minimum_score)
        normalized_dictionary[isbn_key] = normalized_score
        
    return normalized_dictionary

def get_final_hybrid_score(element: tuple) -> float:
    return element[1]

def generate_weighted_hybrid_recommendations(
    target_isbn: str, 
    item_similarity_dataframe: pd.DataFrame, 
    tfidf_matrix, 
    books_dataframe: pd.DataFrame, 
    recommendation_count: int = 5,
    collaborative_weight: float = 0.65,
    content_weight: float = 0.35
) -> list:
    
    if not target_isbn:
        raise ValueError("The target ISBN string cannot be empty.")
        
    if recommendation_count <= 0:
        raise ValueError("The recommendation count must be strictly greater than zero.")

    collaborative_scores_raw = {}
    try:
        collaborative_scores_raw = get_collaborative_scores(item_similarity_dataframe, target_isbn, pool_size=50)
    except KeyError:
        pass
        
    content_scores_raw = {}
    try:
        content_scores_raw = get_content_scores(target_isbn, tfidf_matrix, books_dataframe, pool_size=50)
    except ValueError:
        pass
        
    if not collaborative_scores_raw and not content_scores_raw:
        raise ValueError(f"Could not generate any recommendations for the ISBN {target_isbn}.")

    normalized_collaborative = normalize_scores(collaborative_scores_raw)
    normalized_content = normalize_scores(content_scores_raw)
    
    all_candidate_isbns = set(normalized_collaborative.keys()).union(set(normalized_content.keys()))
    
    final_scores_list = []
    
    for candidate_isbn in all_candidate_isbns:
        collab_score = normalized_collaborative.get(candidate_isbn, 0.0)
        content_score = normalized_content.get(candidate_isbn, 0.0)
        
        weighted_total = (collab_score * collaborative_weight) + (content_score * content_weight)
        final_scores_list.append((candidate_isbn, weighted_total))
        
    sorted_final_scores = sorted(final_scores_list, key=get_final_hybrid_score, reverse=True)
    
    final_recommendations_list = []
    for index_value in range(min(recommendation_count, len(sorted_final_scores))):
        final_recommendations_list.append(sorted_final_scores[index_value][0])

    return final_recommendations_list