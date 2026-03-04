import pandas as pd

def search_books_by_text(query_string: str, books_dataframe: pd.DataFrame, maximum_results: int = 10) -> pd.DataFrame:
    if not query_string or str(query_string).strip() == "":
        raise ValueError("Search query cannot be empty.")
        
    query_lower = query_string.lower()
    
    title_matches = books_dataframe['Book-Title'].astype(str).str.lower().str.contains(query_lower, na=False)
    author_matches = books_dataframe['Book-Author'].astype(str).str.lower().str.contains(query_lower, na=False)
    
    combined_matches = books_dataframe[title_matches | author_matches]
    
    return combined_matches.head(maximum_results)