from google_play_scraper import app, Sort, reviews_all
from app_store_scraper import AppStore
from app_store_scraper import AppStore
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
import streamlit as st
import altair as alt
import joblib
import plotly.graph_objects as go
from func import analyze_reviews
from version import version_info


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Streamlit app to be full screen
# st.set_page_config(layout="wide")

st.markdown('<h1 style="font-size: 70px; color: #1C4CD6;"> HDFC Sky </h1>',
            unsafe_allow_html=True)
st.markdown('<h1 style="font-size: 70px; color: #9347ED;">Customer Review Analytics</h1>',
            unsafe_allow_html=True)


@st.cache_resource(ttl=86400)
def get_reviews():

    a_reviews = AppStore('in', 'hdfc-sky-invest-trade-learn', '6447300262')
    a_reviews.review()
    a_df = pd.DataFrame(np.array(a_reviews.reviews), columns=['review'])
    a_df2 = a_df.join(pd.DataFrame(a_df.pop('review').tolist()))

    a_df2.drop(columns={'isEdited'}, inplace=True)
    a_df2.insert(loc=0, column='source', value='App Store')
    a_df2.rename(columns={'review': 'review_description', 'userName': 'user_name', 'date': 'review_date',
                          'title': 'review_title', 'developerResponse': 'developer_response'}, inplace=True)
    a_df2 = a_df2.where(pd.notnull(a_df2), None)
    a_df2.fillna("Apple App Store", inplace=True)
    df = a_df2
    df = df.sort_values('review_date', ascending=False)
    # df.drop('developer_response', axis=1, inplace=True)
    df.drop('review_title', axis=1, inplace=True)
    return df


df = get_reviews()

stop_words = {'hdfc', 'securities', 'sky', 'please',
              'able', 'bank', 'app', 'good', 'nice', 'best'}
analyze_reviews(df, stop_words)

st.divider()
version_info(
    'https://apps.apple.com/us/app/hdfc-sky-invest-trade-learn/id6447300262', 11)
