import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Connect to the SQLite database
conn = sqlite3.connect('amazon.db')

# Load the dataset from the SQLite database
df = pd.read_sql('SELECT * FROM amazon_data', conn)
conn.close()

# Preprocess the dataset
df = df.dropna()
df = df.drop_duplicates()

# Convert relevant columns to numeric
df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# Clean and preprocess text columns
df['product_name'] = df['product_name'].apply(clean_text)
df['about_product'] = df['about_product'].apply(clean_text)
df['review_content'] = df['review_content'].apply(clean_text)
df['category'] = df['category'].apply(lambda x: x.split('|')[0] if pd.notnull(x) else x)

# Feature Engineering
df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df['review_content']
df['combined_text'] = df['combined_text'].fillna('')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Create a product-user matrix
product_user_matrix = df.pivot_table(index='product_id', values='rating', aggfunc='mean')
product_user_matrix = product_user_matrix.fillna(product_user_matrix.mean())

# Function for hybrid recommendation
def hybrid_recommendation(product_id, content_sim_matrix, product_user_matrix, products, top_n=5):
    idx = products.index[products['product_id'] == product_id][0]
    sim_scores = list(enumerate(content_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    
    if product_id in product_user_matrix.index:
        current_product_rating = product_user_matrix.loc[product_id].values[0]
        similar_rating_products = product_user_matrix.iloc[(product_user_matrix['rating']-current_product_rating).abs().argsort()[:top_n]]
    
        collaborative_recommendations_idx = similar_rating_products.index
        collaborative_recommendations_idx = [products.index[products['product_id'] == pid].tolist()[0] for pid in collaborative_recommendations_idx]
    
    combined_indices = list(set(content_recommendations_idx + collaborative_recommendations_idx))
    recommended_products = products.iloc[combined_indices].copy()
    recommended_products = recommended_products[['product_id', 'product_name', 'about_product', 'rating', 'discount_percentage', 'product_link', 'img_link']]
    
    return recommended_products.head(top_n)

# Streamlit UI
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .title {
        color: #0073e6;
        font-size: 2em;
        font-weight: bold;
    }
    .recommendation {
        border: 2px solid #0073e6;
        padding: 10px;
        border-radius: 5px;
        background-color: #e6f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Amazon Product Recommendation System</h1>", unsafe_allow_html=True)

# User input for product ID
product_id = st.text_input("Enter Product ID:", "")

if product_id:
    if product_id in df['product_id'].values:
        # Display the details of the entered product
        selected_product = df[df['product_id'] == product_id].iloc[0]
        
        st.write(f"**Product Name**: {selected_product['product_name']}")
        st.write(f"**About Product**: {selected_product['about_product']}")
        st.write(f"**Rating**: {selected_product['rating']}")
        st.write(f"**Discount**: {selected_product['discount_percentage']}%")
        st.write(f"**Price**: ₹{selected_product['actual_price']}")
        if selected_product['img_link']:
            st.image(selected_product['img_link'], width=150)
        st.write(f"[Product Link]({selected_product['product_link']})")
        st.write("---")
        
        # Display heading for recommended products
        st.subheader("Recommended Products:")
        
        # Generate and display recommendations
        recommended_products = hybrid_recommendation(product_id, cosine_sim, product_user_matrix, df)
        
        # Display recommended products with links in a single column
        for idx, row in recommended_products.iterrows():
            st.write(f"**{row['product_name']}**")  # Display 'product_name' for recommended product
            st.write(f"**{row['about_product']}**")  # Display 'about_product' for recommended product
            st.write(f"Rating: {row['rating']}")
            st.write(f"Discount: {row['discount_percentage']}%")
            if row['img_link']:
                st.image(row['img_link'], width=150)
            st.write(f"[Product Link]({row['product_link']})")
            st.write("---")
        
    else:
        st.write("Product ID not found in the dataset.")

# Top 10 Most Popular Products
if st.checkbox("Show Top 10 Most Popular Products"):
    top_selling_products = df.sort_values(by='rating_count', ascending=False).head(10)
    
    # Display Top 10 products with images and links
    for idx, row in top_selling_products.iterrows():
        st.write(f"**{row['product_name']}**")  # Display 'product_name' for top-selling products
        st.write(f"**{row['about_product']}**")  # Display 'about_product' for top-selling products
        st.write(f"Rating: {row['rating']}")
        st.write(f"Sales: {row['rating_count']}")
        st.write(f"Discount: {row['discount_percentage']}%")
        if row['img_link']:
            st.image(row['img_link'], width=150)
        st.write(f"[Product Link]({row['product_link']})")
        st.write("---")
        
    # Visualization
    st.subheader("Sales and Rating Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['rating_count'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Rating Counts')
    st.pyplot(fig)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
# from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
# import seaborn as sns
# # from transformers import TFT5ForConditionalGeneration, T5Tokenizer
# from transformers import AutoTokenizer, TFT5ForConditionalGeneration

# # Define the model name
# model_name = "google-t5/t5-small"  # You can change this to any other T5 model like 't5-base' or 't5-large'

# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = TFT5ForConditionalGeneration.from_pretrained(model_name)


# # # Initialize T5 model and tokenizer
# # model_name = "t5-small"  # or you can use 't5-base', 't5-large' for better results
# # model = TFT5ForConditionalGeneration.from_pretrained(model_name)
# # tokenizer = T5Tokenizer.from_pretrained(model_name)

# # Function to clean and preprocess text
# # def clean_text(text):
# #     text = text.lower()
# #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
# #     stop_words = set(stopwords.words('english'))
# #     text = ' '.join([word for word in text.split() if word not in stop_words])
# #     return text

# # Function to generate text using T5
# def generate_t5_text(input_text, task="summarize"):
#     # Define the prefix for T5 task
#     input_text = task + ": " + input_text
#     # Tokenize the input text
#     inputs = tokenizer.encode(input_text, return_tensors="tf", max_length=512, truncation=True)
#     # Generate the output
#     outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
#     # Decode the output text
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Connect to the SQLite database
# conn = sqlite3.connect('amazon.db')

# # Load the dataset from the SQLite database
# df = pd.read_sql('SELECT * FROM amazon_data', conn)
# conn.close()

# # Preprocess the dataset
# df = df.dropna()
# df = df.drop_duplicates()

# # Convert relevant columns to numeric
# df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
# df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
# df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
# df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
# df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# # # Clean and preprocess text columns
# # df['product_name'] = df['product_name'].apply(clean_text)
# # df['about_product'] = df['about_product'].apply(clean_text)
# # df['review_content'] = df['review_content'].apply(clean_text)
# # df['category'] = df['category'].apply(lambda x: x.split('|')[0] if pd.notnull(x) else x)

# # Feature Engineering
# df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df['review_content']
# df['combined_text'] = df['combined_text'].fillna('')

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
# tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# # Compute cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Create a product-user matrix
# product_user_matrix = df.pivot_table(index='product_id', values='rating', aggfunc='mean')
# product_user_matrix = product_user_matrix.fillna(product_user_matrix.mean())

# # Function for hybrid recommendation
# def hybrid_recommendation(product_id, content_sim_matrix, product_user_matrix, products, top_n=5):
#     idx = products.index[products['product_id'] == product_id][0]
#     sim_scores = list(enumerate(content_sim_matrix[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    
#     if product_id in product_user_matrix.index:
#         current_product_rating = product_user_matrix.loc[product_id].values[0]
#         similar_rating_products = product_user_matrix.iloc[(product_user_matrix['rating']-current_product_rating).abs().argsort()[:top_n]]
    
#         collaborative_recommendations_idx = similar_rating_products.index
#         collaborative_recommendations_idx = [products.index[products['product_id'] == pid].tolist()[0] for pid in collaborative_recommendations_idx]
    
#     combined_indices = list(set(content_recommendations_idx + collaborative_recommendations_idx))
#     recommended_products = products.iloc[combined_indices].copy()
#     recommended_products = recommended_products[['product_id', 'about_product', 'rating', 'discount_percentage', 'product_link', 'img_link']]
    
#     return recommended_products.head(top_n)


# # Streamlit UI
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background-color: #f0f0f0;
#     }
#     .sidebar .sidebar-content {
#         background-color: #ffffff;
#     }
#     .title {
#         color: #0073e6;
#         font-size: 2em;
#         font-weight: bold;
#     }
#     .recommendation {
#         border: 2px solid #0073e6;
#         padding: 10px;
#         border-radius: 5px;
#         background-color: #e6f7ff;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown("<h1 class='title'>Amazon Product Recommendation System</h1>", unsafe_allow_html=True)

# # User input for product ID
# product_id = st.text_input("Enter Product ID:", "")

# if product_id:
#     if product_id in df['product_id'].values:
#         # Display the details of the entered product
#         selected_product = df[df['product_id'] == product_id].iloc[0]
        
#         # Use T5 model to generate product name and description text
#         product_name_generated = generate_t5_text(selected_product['product_name'], task="summarize")
#         about_product_generated = generate_t5_text(selected_product['about_product'], task="summarize")
        
#         st.write(f"**Product Name**: {product_name_generated}")
#         st.write(f"**About Product**: {about_product_generated}")
#         st.write(f"**Rating**: {selected_product['rating']}")
#         st.write(f"**Discount**: {selected_product['discount_percentage']}%")
#         st.write(f"**Price**: ₹{selected_product['actual_price']}")
#         if selected_product['img_link']:
#             st.image(selected_product['img_link'], width=150)
#         st.write(f"[Product Link]({selected_product['product_link']})")
#         st.write("---")
        
#         # Display recommendations
#         st.subheader("Recommendation:") 
#         recommended_products = hybrid_recommendation(product_id, cosine_sim, product_user_matrix, df)

#         # Display recommended products with links in a single column
#         for idx, row in recommended_products.iterrows():
#             if product_id in df['product_id'].values:
#             # Display the details of the entered product
#                 select_product = df[df['product_id'] == product_id].iloc[0]
        
#             # Use T5 model to generate product name and description text
#             product_name_generated = generate_t5_text(selected_product['product_name'], task="summarize")
#             about_product_generated = generate_t5_text(selected_product['about_product'], task="summarize")
    
    
#     # Display the generated summaries
#     st.write(f"**{product_name_generated}**")
#     st.write(f"**{about_product_generated}**")
#     st.write(f"Rating: {row['rating']}")
#     st.write(f"Discount: {row['discount_percentage']}%")
#     if row['img_link']:
#         st.image(row['img_link'], width=150)
#     st.write(f"[Product Link]({row['product_link']})")
#     st.write("---")

#       # Display recommendations
# #         st.subheader("Recommendation:")
# #         recommended_products = hybrid_recommendation(product_id, cosine_sim, product_user_matrix, df)
        
# #         # Display recommended products with links in a single column
# #         for idx, row in recommended_products.iterrows():
# #             st.write(f"**{row['product_name']}**")
# #             #the above line shouldnt, be displayed, this product name must be given as input to tf and output from there must be generated. replace it
# #             st.write(f"**{row['about_product']}**")
# #             #the above line shouldnt, be displayed, this product description must be given as input to tf and output from there must be generated. replace it  
# #             st.write(f"Rating: {row['rating']}")
# #             st.write(f"Discount: {row['discount_percentage']}%")
# #             if row['img_link']:
# #                 st.image(row['img_link'], width=150)
# #             st.write(f"[Product Link]({row['product_link']})")
# #             st.write("---")
        

# # Top 10 Most Popular Products
# if st.checkbox("Show Top 10 Most Popular Products"):
#     top_selling_products = df.sort_values(by='rating_count', ascending=False).head(10)
    
#     # Display Top 10 products with images and links
#     for idx, row in top_selling_products.iterrows():
#         about_product_generated = generate_t5_text(row['about_product'], task="summarize")
#         st.write(f"**{about_product_generated}**")
#         st.write(f"Rating: {row['rating']}")
#         st.write(f"Sales: {row['rating_count']}")
#         st.write(f"Discount: {row['discount_percentage']}%")
#         if row['img_link']:
#             st.image(row['img_link'], width=150)
#         st.write(f"[Product Link]({row['product_link']})")
#         st.write("---")
        
#     # Visualization
#     st.subheader("Sales and Rating Distribution")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.histplot(df['rating_count'], bins=20, kde=True, ax=ax)
#     ax.set_title('Distribution of Rating Counts')
#     st.pyplot(fig)
# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
# from nltk.corpus import stopwords
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Function to clean and preprocess text
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     stop_words = set(stopwords.words('english'))
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     return text

# # Connect to the SQLite database
# conn = sqlite3.connect('amazon.db')

# # Load the dataset from the SQLite database
# df = pd.read_sql('SELECT * FROM amazon_data', conn)
# conn.close()

# # Preprocess the dataset
# df = df.dropna()
# df = df.drop_duplicates()

# # Convert relevant columns to numeric
# df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
# df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
# df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
# df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
# df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# # Clean and preprocess text columns
# df['product_name'] = df['product_name'].apply(clean_text)
# df['about_product'] = df['about_product'].apply(clean_text)
# df['review_content'] = df['review_content'].apply(clean_text)
# df['category'] = df['category'].apply(lambda x: x.split('|')[0] if pd.notnull(x) else x)

# # Feature Engineering
# df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df['review_content']
# df['combined_text'] = df['combined_text'].fillna('')

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
# tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# # Compute cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Create a product-user matrix
# product_user_matrix = df.pivot_table(index='product_id', values='rating', aggfunc='mean')
# product_user_matrix = product_user_matrix.fillna(product_user_matrix.mean())

# # Function for hybrid recommendation
# def hybrid_recommendation(product_id, content_sim_matrix, product_user_matrix, products, top_n=5):
#     idx = products.index[products['product_id'] == product_id][0]
#     sim_scores = list(enumerate(content_sim_matrix[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    
#     if product_id in product_user_matrix.index:
#         current_product_rating = product_user_matrix.loc[product_id].values[0]
#         similar_rating_products = product_user_matrix.iloc[(product_user_matrix['rating']-current_product_rating).abs().argsort()[:top_n]]
    
#         collaborative_recommendations_idx = similar_rating_products.index
#         collaborative_recommendations_idx = [products.index[products['product_id'] == pid].tolist()[0] for pid in collaborative_recommendations_idx]
    
#     combined_indices = list(set(content_recommendations_idx + collaborative_recommendations_idx))
#     recommended_products = products.iloc[combined_indices].copy()
#     recommended_products = recommended_products[['product_id', 'product_name', 'about_product', 'rating', 'discount_percentage', 'product_link', 'img_link']]
    
#     return recommended_products.head(top_n)

# # Streamlit UI
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background-color: #f0f0f0;
#     }
#     .sidebar .sidebar-content {
#         background-color: #ffffff;
#     }
#     .title {
#         color: #0073e6;
#         font-size: 2em;
#         font-weight: bold;
#     }
#     .recommendation {
#         border: 2px solid #0073e6;
#         padding: 10px;
#         border-radius: 5px;
#         background-color: #e6f7ff;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown("<h1 class='title'>Amazon Product Recommendation System</h1>", unsafe_allow_html=True)

# # User input for product ID
# product_id = st.text_input("Enter Product ID:", "")

# if product_id:
#     if product_id in df['product_id'].values:
#         # Display the details of the entered product
#         selected_product = df[df['product_id'] == product_id].iloc[0]
        
#         st.write(f"**Product Name**: {selected_product['product_name']}")
#         st.write(f"**About Product**: {selected_product['about_product']}")
#         st.write(f"**Rating**: {selected_product['rating']}")
#         st.write(f"**Discount**: {selected_product['discount_percentage']}%")
#         st.write(f"**Price**: ₹{selected_product['actual_price']}")
#         if selected_product['img_link']:
#             st.image(selected_product['img_link'], width=150)
#         st.write(f"[Product Link]({selected_product['product_link']})")
#         st.write("---")
        
#         # Generate and display recommendations
#         recommended_products = hybrid_recommendation(product_id, cosine_sim, product_user_matrix, df)
        
#         # Display recommended products with links in a single column
#         for idx, row in recommended_products.iterrows():
#             st.write(f"**{row['product_name']}**")  # Display 'product_name' for recommended product
#             st.write(f"**{row['about_product']}**")  # Display 'about_product' for recommended product
#             st.write(f"Rating: {row['rating']}")
#             st.write(f"Discount: {row['discount_percentage']}%")
#             if row['img_link']:
#                 st.image(row['img_link'], width=150)
#             st.write(f"[Product Link]({row['product_link']})")
#             st.write("---")
        
#     else:
#         st.write("Product ID not found in the dataset.")

# # Top 10 Most Popular Products
# if st.checkbox("Show Top 10 Most Popular Products"):
#     top_selling_products = df.sort_values(by='rating_count', ascending=False).head(10)
    
#     # Display Top 10 products with images and links
#     for idx, row in top_selling_products.iterrows():
#         st.write(f"**{row['product_name']}**")  # Display 'product_name' for top-selling products
#         st.write(f"**{row['about_product']}**")  # Display 'about_product' for top-selling products
#         st.write(f"Rating: {row['rating']}")
#         st.write(f"Sales: {row['rating_count']}")
#         st.write(f"Discount: {row['discount_percentage']}%")
#         if row['img_link']:
#             st.image(row['img_link'], width=150)
#         st.write(f"[Product Link]({row['product_link']})")
#         st.write("---")
        
#     # Visualization
#     st.subheader("Sales and Rating Distribution")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.histplot(df['rating_count'], bins=20, kde=True, ax=ax)
#     ax.set_title('Distribution of Rating Counts')
#     st.pyplot(fig)

