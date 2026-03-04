import os
import pandas as pd

def validate_file_existence(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required data file is missing: {file_path}")

def load_ratings_data(directory_path: str) -> pd.DataFrame:
    file_path = os.path.join(directory_path, "BX-Book-Ratings.csv")
    validate_file_existence(file_path)
    
    ratings_dataframe = pd.read_csv(
        file_path, 
        sep=';', 
        encoding='latin-1', 
        escapechar='\\'
    )
    
    expected_columns = ["User-ID", "ISBN", "Book-Rating"]
    for column_name in expected_columns:
        if column_name not in ratings_dataframe.columns:
            raise ValueError(f"Missing required column {column_name} in ratings data.")
            
    return ratings_dataframe

def load_books_data(directory_path: str) -> pd.DataFrame:
    file_path = os.path.join(directory_path, "BX-Books.csv")
    validate_file_existence(file_path)
    
    books_dataframe = pd.read_csv(
        file_path, 
        sep=';', 
        encoding='latin-1', 
        escapechar='\\',
        low_memory=False
    )
    
    expected_columns = [
        "ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", 
        "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"
    ]
    
    for column_name in expected_columns:
        if column_name not in books_dataframe.columns:
            raise ValueError(f"Missing required column {column_name} in books data.")
            
    return books_dataframe

def load_users_data(directory_path: str) -> pd.DataFrame:
    file_path = os.path.join(directory_path, "BX-Users.csv")
    validate_file_existence(file_path)
    
    users_dataframe = pd.read_csv(
        file_path, 
        sep=';', 
        encoding='latin-1', 
        escapechar='\\'
    )
    
    expected_columns = ["User-ID", "Location", "Age"]
    for column_name in expected_columns:
        if column_name not in users_dataframe.columns:
            raise ValueError(f"Missing required column {column_name} in users data.")
            
    return users_dataframe

def initialize_dataset(directory_path: str) -> tuple:
    ratings_dataframe = load_ratings_data(directory_path)
    books_dataframe = load_books_data(directory_path)
    users_dataframe = load_users_data(directory_path)
    
    return ratings_dataframe, books_dataframe, users_dataframe