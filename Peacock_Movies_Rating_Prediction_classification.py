import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import warnings
import streamlit as st
from category_encoders import LeaveOneOutEncoder
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')
all_stopwords = stopwords.words("english")

class data_prep():
    def __init__(self, datafile):
        self.data = datafile
        self.data[['startYear','runtimeMinutes']] = self.data[['startYear','runtimeMinutes']].astype(int)
    
    def genres_encoding(self):
        """
        Encodes the 'genres' column in the movie data.

        Returns
        -------
        None
        """
        def gen_enc(data):
            # Tokenize the 'genres' column and create a list of tokens
            tokens = ' '.join(data['genres']).lower()
            tokens = word_tokenize(tokens)
            tokens = [token for token in tokens if any(char.isalpha() for char in token)]

            # Remove duplicate tokens
            unique_tokens = list(set(tokens))
            
            # Create a DataFrame to store the encoded genres
            encoded_genres = pd.DataFrame()

            # Iterate through each unique token
            for token in unique_tokens:
                # Create a new column for the encoded token
                encoded_genres['type'+token] = data['genres'].apply(lambda x: int(token in x.lower()))

            # Print the encoded genres
            data = pd.concat([data.drop('genres',axis=1), encoded_genres], axis=1)
            return data
        
        # Save the X_train columns as a pkl file
        with open('X_train_columns.pkl', 'rb') as f:
            genres_columns = pickle.load(f)
        
        self.data = gen_enc(self.data).reindex(columns=genres_columns, fill_value=0)

    def text_corpus_column(self):
        self.data['corpus'] = self.data['primaryTitle'] + ' ' + self.data['narrative']
        self.data = self.data.drop(['tconst','primaryTitle','narrative', 'numVotes'],axis=1)     

    def custom_preprocessor(self, text):
        text = re.sub(r"<br\s*/?>\s*<br\s*/?>", " ", text)
        text = re.sub(r"[^\s\w]|_", " ", text)
        text = re.sub(r'[0-9]+', '', text)
        text= text.split()
        text = [lemmatizer.lemmatize(word) for word in text if not word in set(all_stopwords)]
        text = ' '.join(text)
        return text.lower()

    def tfidf_vectorizer(self):

        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        self.tfidf_new = vectorizer.transform(self.data['corpus'])
        self.tfidf_new = pd.DataFrame(self.tfidf_new.toarray(), columns=vectorizer.get_feature_names_out())
        self.tfidf_new.index = self.data.index
    
    def combine_features(self):
        self.data = pd.concat([self.data.drop('corpus',axis=1), self.tfidf_new], axis = 1)
        self.data_cat = self.data.select_dtypes(exclude=np.number).apply(lambda x: x.str.lower())
        self.data_num = self.data.select_dtypes(include=np.number)

    def ohe_cat_cols(self):
        with open('leave_one_out_vectorizer.pkl', 'rb') as f:
            categorical_pipeline = pickle.load(f)
        self.data_cat_encoded = pd.DataFrame(categorical_pipeline.transform(self.data_cat))
        self.data_cat_encoded.columns = categorical_pipeline.named_steps['encoder'].get_feature_names_out()
        self.data = pd.concat([self.data_num.reset_index(),self.data_cat_encoded.reset_index()], axis=1)
        return self.data

    def printer(self):
        display(self.data)

class Model(data_prep):
    def __init__(self, datafile):
        super().__init__(datafile)
        super().genres_encoding()
        super().text_corpus_column()
        super().tfidf_vectorizer()
        super().combine_features()
        super().ohe_cat_cols()
        self.data = self.data.drop('index',axis=1)

    def load_model(self):
        with open('rfc.pkl', 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(self.data)
        return predictions


if __name__ == '__main__':
    st.title('Peacock Film Finance')
    st.subheader('Movie Rating Prediction')
    primaryTitle = st.text_input('Movie Title', placeholder='Enter the movie title...')
    startYear = st.text_input('Year of Release', placeholder='Enter the year of release...')
    runtimeMinutes = st.text_input('Runtime (in minutes)', placeholder='Enter movie runtime in minutes...')
    genres = st.text_input('Genres', placeholder='Enter movie genre...')
    narrative = st.text_input('Plot Summary', placeholder='Enter a brief narrative of the movie plot...')
    director_name = st.text_input('Director', placeholder='Enter name of director...')
    lead_actor_or_actress = st.text_input('Lead Actor/Actress', placeholder='Enter name of lead actor or actress...')

    data = {'primaryTitle': [primaryTitle],
            'startYear': [startYear],
            'runtimeMinutes': [runtimeMinutes],
            'genres': [genres],
            'narrative': [narrative],
            'director_name': [director_name],
            'lead_actor_or_actress': [lead_actor_or_actress]}

    df = pd.DataFrame(data)

    if st.button('Predict'):
        m = Model(df)
        if m.load_model() == 0:
            st.markdown('The predicted rating is :red[not expected] to do well with rating at :red[lesser than 7 out of 10]')
        else:
            st.markdown('The predicted rating is :green[expected] to do well with rating at :green[7 or above out of 10]')
