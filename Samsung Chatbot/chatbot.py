import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

dt= pd.read_csv('Samsung Dialog.txt', sep= ':', header= None)
dt.head()

cust = dt.loc[dt[0] == 'Customer']
sales = dt.loc[dt[0] == 'Sales Agent']

dt = pd.DataFrame()
dt['Questions'] = cust[1].reset_index(drop = True)
dt['Answer'] = sales[1].reset_index(drop = True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        # The code above does the following:
        # Identifies every word in the sentence
        # Turns it to a lower case
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)

dt['tokenized Questions'] = dt['Questions'].apply(preprocess_text)

corpus = dt['tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()
vectorised_corpus = tfidf_vectorizer.fit_transform(corpus)

def get_response(user_input):
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, vectorised_corpus) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return dt['Answer'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response


# --------------------- STREAMLIT IMPLEMENTATION  ------------
st.markdown("<h1 style = 'text-align: center; color: #176B87'>SAMSUNG CUSTOMER CARE</h1>", unsafe_allow_html = True)
st.markdown("<h6 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY AyoBankole Mushin DataGuy</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)
col1, col2 = st.columns(2)
col1.image('chatbot.png.png', caption = 'Ask Samsung related Questions')


def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vectorizer.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, vectorised_corpus)
    most_similar_index = most_similar.argmax()
    
    return dt['Answer'].iloc[most_similar_index]

chatbot_greeting = [
    "Hello there, welcome to Samsung Customer Care. pls ejoy your usage",
    "Hi user, This bot is created by AyoBankole The Mushin DataGuy, enjoy your usage",
    "Hi hi, How you?",
    "Broda, Abeg enjoy your usage",
    "Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']


user_q = col2.text_input('Pls ask your Samsung Galaxy related question: ')
if user_q in user_greeting:
    col2.write(random.choice(chatbot_greeting))
elif user_q in exit_word:
    col2.write('Thank you for your usage. Bye')
elif user_q == '':
    st.write('')
else:
    responses = bot_response(user_q)
    col2.write(f'ChatBot:{responses}')