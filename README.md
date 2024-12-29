# Summer-Prod-recommendation

This repository contains the code for an **Amazon Product Recommendation System**, which is hosted on **Streamlit Community Cloud**. The application provides personalized product recommendations based on both content similarity and user ratings.

## Features

- **Hybrid Recommendation System**: Combines content-based filtering and collaborative filtering.
- **Product Details**: Displays details of the searched product, including name, description, rating, discount, price, and image.
- **Recommended Products**: Suggests similar products to the user.
- **Top 10 Most Popular Products**: Showcases the most popular products based on sales and ratings.
- **Visualizations**: Includes a distribution of rating counts for better insights.

## How to Run Locally

### Prerequisites
- Python 3.7 or later
- Streamlit
- Pandas
- NumPy
- SQLite3
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/amazon-product-recommendation.git
    cd amazon-product-recommendation
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure the `amazon.db` SQLite database file is present in the project directory.

4. Run the application:
    ```bash
    streamlit run app.py
    ```

5. Open the app in your browser by navigating to the URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment
The app is deployed on **Streamlit Community Cloud**. .

To deploy your version:
1. Create a Streamlit account and connect it to your GitHub.
2. Push the repository to GitHub.
3. In Streamlit, create a new app, select your repository, and configure the branch and main file (`app.py`).
4. Deploy!

## Dataset
The app fetches data from an SQLite database (`amazon.db`). The database should include a table named `amazon_data` with the following columns:
- `product_id`
- `product_name`
- `about_product`
- `rating`
- `rating_count`
- `discounted_price`
- `actual_price`
- `discount_percentage`
- `product_link`
- `img_link`
- `category`
- `review_content`

## Key Libraries Used
- **Streamlit**: For building the web interface.
- **Pandas**: Data manipulation and cleaning.
- **SQLite3**: Database connection and querying.
- **Scikit-learn**: TF-IDF vectorization and cosine similarity calculation.
- **NLTK**: Text cleaning and preprocessing.
- **Matplotlib & Seaborn**: Data visualization.

## Usage
1. Enter a **Product ID** in the input box to view product details and get recommendations.
2. Check the **"Show Top 10 Most Popular Products"** box to view popular products and their distribution.


