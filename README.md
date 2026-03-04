# Inkwell
![GitHub Created At](https://img.shields.io/github/created-at/Stonky-Boi/Inkwell)
![GitHub contributors](https://img.shields.io/github/contributors/Stonky-Boi/Inkwell)
![GitHub License](https://img.shields.io/github/license/Stonky-Boi/Inkwell)

InkWell is a robust book recommendation engine that combines item-based collaborative filtering (using K-Nearest Neighbors) and content-based filtering (using TF-IDF and Cosine Similarity). Built entirely in Python, it features strict error handling, explicit data validation, and an interactive Streamlit web interface.

## Project Structure

```text
inkwell/
├── data/                       # Directory for dataset CSV files
├── recommender/                # Core recommendation engine modules
│   ├── __init__.py
│   ├── data_loader.py          # Strict dataset ingestion and validation
│   ├── collaborative_filter.py # KNN-based user-item interaction modeling
│   ├── content_filter.py       # TF-IDF metadata vectorization 
│   └── hybrid_engine.py        # Interleaved recommendation aggregation
├── app.py                      # Streamlit frontend application
├── requirements.txt            # Python dependencies
├── download_dataset.sh         # Shell script to fetch Book-Crossing dataset
└── README.md
```

## Prerequisites

* Python 3.8 or higher.
* Recommended: A virtual environment to isolate dependencies.

## Installation and Setup

**1. Install Dependencies**
Install the required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

**2. Download the Dataset**
The application requires the Book-Crossing dataset (`BX-Book-Ratings.csv`, `BX-Books.csv`, and `BX-Users.csv`) to function. You must make the provided shell script executable and run it to populate the `data/` directory.

Execute the following commands in your terminal:

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

*Note: The application will explicitly throw a `FileNotFoundError` upon startup if these files are missing.*

**3. Launch the Application**
Start the Streamlit web server:

```bash
streamlit run app.py
```

## Usage

1. Open your web browser to the local URL provided by Streamlit.
2. In the sidebar, enter a valid 10-character ISBN (e.g., `034545104X`).
3. Set your desired number of recommendations using the numeric input.
4. Click **Generate Recommendations**. The system will aggregate results from both the collaborative and content-based models, displaying the book covers and metadata.

## Architecture Notes

* **Explicit Error Handling:** All modules actively check for empty dataframes, missing columns, and invalid keys, throwing standard Python exceptions (`ValueError`, `KeyError`) to prevent silent failures.
* **Performance:** Heavy operations, such as TF-IDF matrix computation and KNN model training, are cached in the Streamlit frontend (`@st.cache_resource`) to ensure a responsive user experience.