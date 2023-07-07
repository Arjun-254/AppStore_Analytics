from google_play_scraper import app, Sort, reviews, reviews_all
import pandas as pd
import numpy as np
from datetime import date, datetime
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

# Streamlit app to be full screen
# st.set_page_config(layout="wide")

st.markdown('<h1 style="font-size: 70px; color: #E3142D;"> Upstox </h1>',
            unsafe_allow_html=True)
st.markdown('<h1 style="font-size: 70px; color: #9347ED;">Customer Review Analytics </h1>',
            unsafe_allow_html=True)

stop_words = {'upstox', 'please',
              'able', 'bank', 'app', 'good', 'nice', 'best'}
analyze_reviews('ReviewsUP.csv', stop_words)
