def calculate_precision_at_k(recommended_isbns_list: list, relevant_isbns_set: set, k_value: int) -> float:
    if k_value <= 0:
        raise ValueError("The k_value must be strictly greater than zero.")
        
    if not recommended_isbns_list:
        return 0.0
        
    top_k_recommendations = set(recommended_isbns_list[:k_value])
    true_positives = top_k_recommendations.intersection(relevant_isbns_set)
    
    precision_score = len(true_positives) / float(k_value)
    return precision_score

def calculate_recall_at_k(recommended_isbns_list: list, relevant_isbns_set: set, k_value: int) -> float:
    if k_value <= 0:
        raise ValueError("The k_value must be strictly greater than zero.")
        
    if not relevant_isbns_set:
        raise ValueError("The relevant_isbns_set cannot be empty to calculate recall.")
        
    if not recommended_isbns_list:
        return 0.0
        
    top_k_recommendations = set(recommended_isbns_list[:k_value])
    true_positives = top_k_recommendations.intersection(relevant_isbns_set)
    
    recall_score = len(true_positives) / float(len(relevant_isbns_set))
    return recall_score