import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import ast
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Preprocess_MovieDataset.csv')
# copy data into a new dataframe
new_data = data.copy()
# drop unrequired columns
new_data.drop(['budget', 'revenue', 'production_companies',
       'production_countries', 'runtime', 'spoken_languages', 
       'status', 'tagline', 'vote_average', 'vote_count', 'original_title'], axis=1, inplace= True)

vectorizer = TfidfVectorizer(min_df=3, max_features=None,
                             strip_accents='unicode', analyzer='word',
                             token_pattern=r',', ngram_range=(1,3),
                             stop_words='english')

# ------fill the null values with empty string
new_data['genres'] = new_data['genres'].fillna('')
# --------------fitting the TF-IDF on the genre text
tf_matrix = vectorizer.fit_transform(new_data['genres'])
# --------compute the sigmoid kernel
sig = sigmoid_kernel(tf_matrix, tf_matrix)
# ------------reverse mapping of indexes and movie titles
index = pd.Series(new_data.index, index=new_data['title']).drop_duplicates()

# ---------STREAMLIT IMPLEMENTATION-----------#
st.markdown("<h1 style='text-align: center; color: #35374B;font-family: helvetica '>MOVIE RECOMMENDER SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; margin-top: 0rem; color: #50727B;font-family: cursive'>BUILT BY AyoBankole The Mushin DataGuy</h4>", unsafe_allow_html=True)
st.image('pngwing.com.png', width=250, use_column_width=True)
st.markdown("<h4 style='color: #78A083; text-align: center; font-family: cursive'>Project Overview</h4>", unsafe_allow_html=True)
st.markdown("<p>The Movie Recommendation System project aims to build an intelligent system that suggests movies to users based on their preferences and behaviors. Leveraging the TMDB 5000 Movie Dataset, which contains extensive metadata on thousands of movies, the system will employ various recommendation techniques to provide personalized movie suggestions to users.The Movie Recommendation System project leverages the TMDB 5000 Movie Dataset to build an efficient and personalized movie recommendation platform. By combining collaborative and content-based filtering techniques and deploying a user-friendly interface, the system aims to enhance user engagement and satisfaction by delivering relevant and engaging movie suggestions. </p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5 style='color: #78A083; text-align: center; font-family: cursive'>DataSet</h4>", unsafe_allow_html=True)
st.dataframe(new_data, use_container_width = True )
# Sidebar for user input
st.sidebar.title('Welcome User')
st.sidebar.image('pngwing.com (1).png', width=250, use_column_width=True)
movie_title = st.sidebar.selectbox('Select a movie:', new_data['title'])

# Recommendation algorithm
# Example: Use TF-IDF and cosine similarity for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_data['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend similar movies
def recommendations(title, sig=sig):
    # get index corresponding to original title
    idx = index[title]

    # get the pairwise similarity score
    sig_scores = list(enumerate(sig[idx]))

    # sort the movies
    sig_scores = sorted(sig_scores, key=lambda x:x[1], reverse= True)

    # scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # movie indexes
    movie_index = [i[0] for i in sig_scores]

    # top 5 similar movies
    return new_data['title'].iloc[movie_index]

# Display recommended movies
st.sidebar.subheader('Recommended Movies for You:')
recommended_movies = recommendations(movie_title)  # Assuming movie_title is defined somewhere
for movie in recommended_movies:
    st.sidebar.write(movie)
    # st.sidebar.balloons()

