# Inkwell
![GitHub Created At](https://img.shields.io/github/created-at/Stonky-Boi/Inkwell)
![GitHub contributors](https://img.shields.io/github/contributors/Stonky-Boi/Inkwell)
![GitHub License](https://img.shields.io/github/license/Stonky-Boi/Inkwell)

InkWell is a robust book recommendation engine that combines item-based collaborative filtering (using K-Nearest Neighbors) and content-based filtering (using TF-IDF and Cosine Similarity). Built entirely in Python, it features strict error handling, explicit data validation, and an interactive Streamlit web interface.

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
