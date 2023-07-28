from google_play_scraper import app, Sort, reviews, reviews_all
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


def analyze_reviews(df, custom_stop_words):
    hide = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state['page'] = None
    if (st.session_state['page'] != custom_stop_words):
        st.session_state['filter_pressed'] = False
        st.session_state['start_date'] = datetime.now().date()
        st.session_state['end_date'] = datetime.now().date()
        st.session_state['rating_filter'] = "All"

    st.session_state['run_Model'] = False
    st.session_state['page'] = custom_stop_words

    if 'filter_pressed' not in st.session_state:
        st.session_state['filter_pressed'] = False

    # Check if the start_date key exists in session_state, if not initialize it
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = datetime.now().date()

    # Check if the end_date key exists in session_state, if not initialize it
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = datetime.now().date()

    # Check if the rating_filter key exists in session_state, if not initialize it
    if 'rating_filter' not in st.session_state:
        st.session_state['rating_filter'] = "All"

    # Analytics date window wise
    min_review_date = pd.to_datetime(df['review_date']).min().date()
    st.title('Custom Search by Date Range')
    pd.set_option('display.width', 1000)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            'Select start date', value=st.session_state['start_date'],  min_value=min_review_date, max_value=datetime.now().date())
    with col2:
        end_date = st.date_input(
            'Select end date', value=st.session_state['end_date'], min_value=min_review_date, max_value=datetime.now().date())
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df['review_date'] = pd.to_datetime(df['review_date'])
    dfsearch = df  # to search(full data)
    df = df[(df['review_date'] >= start_date)
            & (df['review_date'] <= end_date)]
    rating_options = st.radio("Select Rating Filter", ["All", "4 and below", "5 only"], index=[
        "All", "4 and below", "5 only"].index(st.session_state['rating_filter']))

    # Date Delta
    delta = end_date-start_date
    day_interval = delta.days+1

    filter_button = st.button('Filter reviews')

    # Check if the button is pressed and the dataframe is not empty
    if filter_button and not df.empty:
        st.session_state['filter_pressed'] = True

    # Save the states of the filters
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['rating_filter'] = rating_options

    # Use the filter state in your application
    if st.session_state['filter_pressed']:
        toggle = st.radio('Select Visualization', [
            'Rating Histogram', 'Rating Pie'])
        ratings = df['rating'].value_counts().sort_index()

        if toggle == 'Rating Histogram':
            fig = go.Figure(data=[go.Bar(x=ratings.index, y=ratings.values)])
            fig.update_layout(
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True)
            )
            st.title('Review Counts by Star Rating')
            st.plotly_chart(
                fig, config={'displayModeBar': False}, use_container_width=True)

        else:
            # Rating Pie
            st.title('Review Counts by Star Rating')
            rating_percentages = ratings / ratings.sum() * 100
            pie_data = pd.DataFrame({
                'Rating': rating_percentages.index,
                'Percentage': rating_percentages.values
            })
            chart = alt.Chart(pie_data).mark_arc().encode(
                theta='Percentage:Q',
                color='Rating:N'
            ).properties(
                width=600,
                height=400
            ).configure_legend(
                title=None,
                orient='top',
                labelFontSize=14,
                titleFontSize=16
            )
            st.altair_chart(chart, use_container_width=True)

        # Review traffic
        df['review_date'] = pd.to_datetime(df['review_date'])
        df.set_index('review_date', inplace=True)
        review_counts = df.resample('D').size()
        st.title('Review Distribution Over Time')
        st.subheader('Number of Reviews')
        st.line_chart(review_counts)

        st.warning(
            "Star and Version filters are applied after this point")

        # Stars-wise Review Filtering
        if rating_options == "All":
            # No rating filter applied
            df = df
        elif rating_options == "4 and below":
            df = df[df["rating"] <= 4]
        else:
            df = df[df["rating"] == 5]

        comments = " ".join(df['review_description'])
        words = word_tokenize(comments)

        def clean_words(new_tokens, custom_stop_words):
            new_tokens = [t.lower() for t in new_tokens]
            stop_words = set(stopwords.words('english'))
            if custom_stop_words:
                stop_words.update(custom_stop_words)
            new_tokens = [t for t in new_tokens if t not in stop_words]
            new_tokens = [t for t in new_tokens if t.isalpha()]
            lemmatizer = WordNetLemmatizer()
            new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
            return new_tokens

        cleaned_words = clean_words(words, custom_stop_words)
        if len(cleaned_words) > 0:
            bow = Counter(cleaned_words)
            bow2 = Counter(ngrams(cleaned_words, 2))
            bow3 = Counter(ngrams(cleaned_words, 3))

            word_freq = pd.DataFrame(bow.items(), columns=['word', 'frequency']).sort_values(
                by='frequency', ascending=False)
            word_pairs = pd.DataFrame(bow2.items(), columns=[
                'pairs', 'frequency']).sort_values(by='frequency', ascending=False)
            trigrams = pd.DataFrame(bow3.items(), columns=[
                                    'trigrams', 'frequency']).sort_values(by='frequency', ascending=False)

            ### STREAMLIT CODE###

            # Words
            word_freq = word_freq[word_freq['word'] != 'app']
            data = word_freq.head(30).sort_values(
                by='frequency', ascending=False)
            if (len(data) > 0):
                st.title('Word Frequency Bar Graph ('+rating_options+')')
                st.subheader('Top 30 Words')
                chart = alt.Chart(data).mark_bar().encode(
                    x='frequency:Q',
                    y=alt.Y('word:N', sort='-x'),
                    color=alt.value('#4491FF')
                ).properties(
                    width=600,
                    height=800
                )
                st.altair_chart(chart, use_container_width=True)

                # Filter the reviews that contain the selected word
                selected_word = st.selectbox(
                    'Select a word:', data['word'].tolist())
                selected_reviews = df[df['review_description'].str.contains(
                    selected_word, case=False)]
                # Display the selected reviews
                st.title(f'Reviews Containing "{selected_word}"')
                st.dataframe(selected_reviews)
            else:
                st.warning("Insufficient Data(Please Change Filters)")
            ################################################

            # PAIRS
            data = word_pairs.head(35)
            st.title('Word Pair Frequency Bar Graph ('+rating_options+')')
            st.subheader('Top 35 Pairs')
            if (len(data) > 0):
                chart = alt.Chart(data).mark_bar().encode(
                    x='frequency:Q',
                    y=alt.Y('pairs:N', sort='-x'),
                    color=alt.value('#5A0BA9')
                ).properties(
                    width=600,
                    height=800
                )
                chart = chart.configure_axis(
                    labelFontSize=16,
                    titleFontSize=20
                )
                st.altair_chart(chart, use_container_width=True)

                selected_pair = st.selectbox(
                    'Select a word pair:', data['pairs'].apply(lambda x: ' '.join(x)).tolist())
                selected_reviews = df[df['review_description'].str.contains(
                    selected_pair, case=False)]
                st.title(f'Reviews Containing "{selected_pair}"')
                st.dataframe(selected_reviews)
            else:
                st.warning("Insufficient Data(Please Change Filters)")
            #############

            # TRIGRAMS
            data = trigrams.head(20)
            st.title(
                'Visualization of Trigram Frequency in Reviews ('+rating_options+')')
            st.subheader('Top 20 Trigrams')
            if (len(data) > 0):
                chart = alt.Chart(data).mark_bar().encode(
                    x='frequency:Q',
                    y=alt.Y('trigrams:N', sort='-x'),
                    color=alt.value('#E466CD')
                ).properties(
                    width=800,
                    height=900
                )
                chart = chart.configure_axis(
                    labelFontSize=12,
                    titleFontSize=18

                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Insufficient Data(Please Change Filters)")

            # ///SEARCH FUNC////
            st.title(f'Custom Search(All Reviews) ')
            search_word = st.text_input('Enter a word to search in reviews')
            selected_reviews = dfsearch[dfsearch['review_description'].str.contains(
                search_word, case=True)]
            selected_reviews = selected_reviews.reset_index(drop=True)
            st.dataframe(selected_reviews)

            # Download Data
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df(dfsearch)

            st.download_button(
                label="Download all available review data as CSV",
                data=csv,
                file_name='Reviews.csv',
                mime='text/csv',
            )

            ########   Sentiment Analysis Start   ##############
            if st.button('Run Sentiment Analytics'):
                st.session_state['run_Model'] = True

            if st.session_state['run_Model'] and (day_interval <= 7):
                with st.spinner("Loading Sentiment Analysis"):
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch

                    tokenizer = AutoTokenizer.from_pretrained(
                        'nlptown/bert-base-multilingual-uncased-sentiment')
                    model = AutoModelForSequenceClassification.from_pretrained(
                        'nlptown/bert-base-multilingual-uncased-sentiment')

                    ratings = []
                    review_lengths = []
                    for val in df['review_description']:
                        tokens = tokenizer.encode(val, return_tensors='pt')
                        result = model(tokens)
                        rating = int(torch.argmax(result.logits))+1
                        ratings.append(rating)
                        review_lengths.append(len(tokens[0]))

                    mean_length = np.mean(review_lengths)
                    mean_rating = np.mean(ratings)
                    mean_ratingrounded = np.round_(mean_rating)
                    actual_mean = df['rating'].mean()

                    sentiment_labels = {
                        1: "Very Unsatisfied",
                        2: "Unsatisfied",
                        3: "Neutral",
                        4: "Satisfied",
                        5: "Very Satisfied"
                    }

                    # Get the sentiment label for the mean rating
                    mean_sentiment = sentiment_labels.get(
                        int(mean_ratingrounded))
                    st.metric(label="Actual Rating Mean(1-5)",
                              value=round(actual_mean, 2))
                    st.metric(label="Mean Sentiment Rating (1-5)",
                              value=round(mean_rating, 2))
                    st.metric(label="Sentiment of Reviews in Selected Timeframe",
                              value=mean_sentiment)
                    st.metric(label="Average Review Length in Selected Timeframe",
                              value=round(mean_length, 2))
                    ########   Sentiment Analysis End   ##############
            elif st.session_state['run_Model'] and (day_interval > 7):
                st.warning(
                    "Sentiment analysis can only be calculated for a time interval of 7 days.")

            del words, cleaned_words, bow, bow2, bow3, word_freq, word_pairs, trigrams
        else:
            st.warning(
                "Please select a different date range to filter the reviews (Not enough data for analysis)")

    else:
        st.warning(
            "Please select a different date range to filter the reviews (Not enough data for analysis)")
