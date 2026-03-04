import os
import streamlit as st
import pandas as pd
from recommender.data_loader import initialize_dataset
from recommender.collaborative_filter import filter_and_merge_data, create_user_item_matrix, train_svd_model
from recommender.content_filter import create_book_features, build_tfidf_feature_matrix
from recommender.hybrid_engine import generate_weighted_hybrid_recommendations
from recommender.search_engine import search_books_by_text
from recommender.evaluation_metrics import calculate_precision_at_k, calculate_recall_at_k

@st.cache_data
def prepare_datasets() -> tuple:
    directory_path = "data"
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The directory '{directory_path}' does not exist.")
        
    ratings_dataframe, books_dataframe, users_dataframe = initialize_dataset(directory_path)
    
    filtered_ratings_dataframe = filter_and_merge_data(ratings_dataframe, users_dataframe)
    user_item_matrix = create_user_item_matrix(filtered_ratings_dataframe)
    books_dataframe_with_features = create_book_features(books_dataframe)
    
    return books_dataframe, user_item_matrix, books_dataframe_with_features, filtered_ratings_dataframe

@st.cache_resource
def initialize_models(user_item_matrix: pd.DataFrame, books_dataframe_with_features: pd.DataFrame) -> tuple:
    if user_item_matrix.empty or books_dataframe_with_features.empty:
        raise ValueError("Cannot initialize models with empty dataframes.")

    item_similarity_dataframe, svd_metrics = train_svd_model(user_item_matrix)
    tfidf_matrix = build_tfidf_feature_matrix(books_dataframe_with_features)
    
    return item_similarity_dataframe, svd_metrics, tfidf_matrix

def display_book_details(isbn_string: str, books_dataframe: pd.DataFrame):
    book_records = books_dataframe[books_dataframe['ISBN'] == isbn_string]
    if book_records.empty:
        st.warning(f"Book details not found for ISBN: {isbn_string}")
        return
        
    book_row = book_records.iloc[0]
    
    st.subheader(book_row['Book-Title'])
    st.write(f"**Author:** {book_row['Book-Author']}")
    st.write(f"**Year:** {book_row['Year-Of-Publication']}")
    st.write(f"**Publisher:** {book_row['Publisher']}")
    
    image_url = book_row['Image-URL-M']
    if pd.notna(image_url) and str(image_url).strip() != "":
        st.image(image_url)

def run_application():
    st.set_page_config(page_title="InkWell Recommender", layout="wide")
    st.title("InkWell Book Recommendation Engine")
    
    try:
        books_dataframe, user_item_matrix, books_dataframe_with_features, filtered_ratings_dataframe = prepare_datasets()
        item_similarity_dataframe, svd_metrics, tfidf_matrix = initialize_models(user_item_matrix, books_dataframe_with_features)
    except Exception as initialization_error:
        st.error(f"Initialization Error: {initialization_error}")
        return

    # Sidebar: Global Metrics and Engine Tuning
    st.sidebar.header("Global Engine Metrics")
    st.sidebar.metric(label="SVD Root Mean Square Error", value=f"{svd_metrics['RMSE']:.4f}")
    st.sidebar.metric(label="SVD Mean Absolute Error", value=f"{svd_metrics['MAE']:.4f}")
    st.sidebar.markdown("---")
    
    st.sidebar.header("Engine Weights")
    collaborative_weight = st.sidebar.slider("Collaborative Weight (SVD):", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
    content_weight = 1.0 - collaborative_weight
    st.sidebar.write(f"Content Weight (TF-IDF): **{content_weight:.2f}**")
    
    recommendation_count = st.sidebar.number_input("Recommendations (K):", min_value=1, max_value=20, value=5)

    # Main Interface Tabs
    tab_search, tab_evaluate = st.tabs(["Book Search", "System Evaluation"])

    with tab_search:
        st.header("Search & Recommend")
        search_query = st.text_input("Search by Book Title or Author:")

        if search_query:
            try:
                search_results = search_books_by_text(search_query, books_dataframe)
                
                if search_results.empty:
                    st.warning("No books found matching your query.")
                else:
                    options_dictionary = {}
                    for index_value, row in search_results.iterrows():
                        display_text = f"{row['Book-Title']} by {row['Book-Author']} ({row['Year-Of-Publication']})"
                        options_dictionary[display_text] = row['ISBN']
                        
                    selected_display_text = st.selectbox("Select a Match:", list(options_dictionary.keys()))
                    target_isbn = options_dictionary[selected_display_text]

                    if st.button("Generate Recommendations", key="btn_search"):
                        st.markdown("---")
                        st.subheader("Recommended Books")
                        
                        recommendations_list = generate_weighted_hybrid_recommendations(
                            target_isbn,
                            item_similarity_dataframe,
                            tfidf_matrix,
                            books_dataframe_with_features,
                            recommendation_count,
                            collaborative_weight,
                            content_weight
                        )
                        
                        columns_list = st.columns(len(recommendations_list))
                        for index_value, recommended_isbn in enumerate(recommendations_list):
                            with columns_list[index_value]:
                                display_book_details(recommended_isbn, books_dataframe)
                                
            except Exception as search_error:
                st.error(f"Search Error: {search_error}")

    with tab_evaluate:
        st.header("User-Centric Ranking Evaluation")
        st.write("Test the engine's precision and recall by selecting a known user and predicting their hidden highly-rated books.")
        
        # Get a list of active users from the matrix to evaluate
        active_users_list = user_item_matrix.index.tolist()
        selected_user_id = st.selectbox("Select a Test User-ID:", active_users_list[:100]) # Limiting to 100 for dropdown performance
        
        if st.button("Run Evaluation", key="btn_eval"):
            try:
                # 1. Fetch user's historical ratings
                user_history = filtered_ratings_dataframe[filtered_ratings_dataframe['User-ID'] == selected_user_id]
                
                # 2. Define "Relevant" items (e.g., books they rated 7 or higher)
                relevant_books = user_history[user_history['Book-Rating'] >= 7.0]
                
                if len(relevant_books) < 2:
                    st.warning(f"User {selected_user_id} does not have enough high-rated books (>=7) to perform a meaningful evaluation.")
                else:
                    # 3. Sort by rating to use their absolute favorite book as the target seed
                    sorted_relevant_books = relevant_books.sort_values(by='Book-Rating', ascending=False)
                    target_seed_isbn = sorted_relevant_books.iloc[0]['ISBN']
                    
                    # 4. The remaining highly-rated books form the ground truth set
                    relevant_isbns_set = set(sorted_relevant_books['ISBN'].values[1:])
                    
                    st.write(f"**Target Seed Book (User's Highest Rated):** {target_seed_isbn}")
                    st.write(f"**Number of Hidden Relevant Books:** {len(relevant_isbns_set)}")
                    
                    # 5. Generate Recommendations
                    recommendations_list = generate_weighted_hybrid_recommendations(
                        target_seed_isbn,
                        item_similarity_dataframe,
                        tfidf_matrix,
                        books_dataframe_with_features,
                        recommendation_count,
                        collaborative_weight,
                        content_weight
                    )
                    
                    # 6. Calculate Metrics
                    precision_score = calculate_precision_at_k(recommendations_list, relevant_isbns_set, recommendation_count)
                    recall_score = calculate_recall_at_k(recommendations_list, relevant_isbns_set, recommendation_count)
                    
                    # Display Results
                    metric_col_1, metric_col_2 = st.columns(2)
                    metric_col_1.metric(label=f"Precision@{recommendation_count}", value=f"{precision_score:.4f}")
                    metric_col_2.metric(label=f"Recall@{recommendation_count}", value=f"{recall_score:.4f}")
                    
                    st.markdown("---")
                    st.subheader("Engine Output")
                    output_columns = st.columns(len(recommendations_list))
                    for index_value, recommended_isbn in enumerate(recommendations_list):
                        with output_columns[index_value]:
                            is_hit = "HIT" if recommended_isbn in relevant_isbns_set else "MISS"
                            st.write(f"**{is_hit}**")
                            display_book_details(recommended_isbn, books_dataframe)
                            
            except Exception as evaluation_error:
                st.error(f"Evaluation Error: {evaluation_error}")

if __name__ == "__main__":
    run_application()
