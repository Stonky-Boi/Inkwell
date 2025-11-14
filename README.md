# Inkwell
![GitHub Created At](https://img.shields.io/github/created-at/Stonky-Boi/Inkwell)
![GitHub contributors](https://img.shields.io/github/contributors/Stonky-Boi/Inkwell)
![GitHub License](https://img.shields.io/github/license/Stonky-Boi/Inkwell)
A machine learning-based book recommender system using:
- **Collaborative Filtering (CF):** Uses user-item interactions (SVD) to find similar users and recommend books.
- **Content-Based Filtering:** Uses TF-IDF and cosine similarity to recommend books similar to those a user likes.
- **Hybrid Approach:** Combines both CF and content-based methods for better recommendations.

🔹 **Tech Stack:** Python, Pandas, Scikit-learn, Streamlit

🔹 **Features:**
- User-based and content-based recommendations
- Interactive Streamlit web app
- Model evaluation using RMSE

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Stonky-Boi/Inkwell.git
   cd Inkwell
   ```
2. Navigate to the project directory and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run ml_app.py
   ```
---