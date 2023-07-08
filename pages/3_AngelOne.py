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

# # Streamlit app to be full screen
# # st.set_page_config(layout="wide")
# st.markdown('<h1 style="font-size: 70px; color: #E3142D;"> Angel One </h1>',
#             unsafe_allow_html=True)
# st.markdown('<h1 style="font-size: 70px; color: #9347ED;">Customer Review Analytics</h1>',
#             unsafe_allow_html=True)

# stop_words = {'angel', 'one', 'please',
#               'able', 'bank', 'app', 'good', 'nice', 'best'}
# analyze_reviews('ReviewsA.csv', stop_words)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google_play_scraper import app, Sort, reviews_all
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

# Streamlit app to be full screen
# st.set_page_config(layout="wide")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Set the cache expiry time to 24 hours
@st.cache_resource(ttl=86400)
def get_reviews():
    MAX_REVIEWS = 30000
    count = 200
    g_reviews = []
    continuation_token = None
    while len(g_reviews) < MAX_REVIEWS:
        reviews_batch, continuation_token = reviews(
            'com.msf.angelmobile',
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=count,
            continuation_token=continuation_token
        )
        g_reviews.extend(reviews_batch)
        if continuation_token is None:
            break

        # Adjust the remaining count based on the already scraped reviews
        remaining_reviews = MAX_REVIEWS - len(g_reviews)
        count = min(remaining_reviews, count)
    return g_reviews


g_reviews = get_reviews()


g_df = pd.DataFrame(np.array(g_reviews), columns=['review'])
g_df2 = g_df.join(pd.DataFrame(g_df.pop('review').tolist()))

g_df2.drop(columns={'userImage', 'reviewCreatedVersion'}, inplace=True)
g_df2.rename(columns={'score': 'rating', 'userName': 'user_name', 'reviewId': 'review_id', 'content': 'review_description', 'at': 'review_date',
             'replyContent': 'developer_response', 'repliedAt': 'developer_response_date', 'thumbsUpCount': 'thumbs_up'}, inplace=True)
g_df2.insert(loc=0, column='source', value='Google Play')
g_df2.insert(loc=3, column='review_title', value=None)


df = g_df2


df.drop('review_title', axis=1, inplace=True)
df.drop('developer_response', axis=1, inplace=True)
df.drop('developer_response_date', axis=1, inplace=True)
df = df[df['review_date'].dt.year == date.today().year]

# To get the first occurence of new app release in reviews to pinpoint update
dfversion = df[['appVersion', 'review_date']]
dfversion = dfversion.drop_duplicates(subset='appVersion', keep='last')
dfversion['review_date'] = pd.to_datetime(
    dfversion['review_date']).dt.strftime('%d/%m/%Y')


st.markdown('<h1 style="font-size: 70px; color: #1C4CD6;"> Angel One </h1>',
            unsafe_allow_html=True)
st.markdown('<h1 style="font-size: 70px; color: #9347ED;">Customer Review Analytics</h1>',
            unsafe_allow_html=True)


st.session_state['run_Model'] = False

if 'page' in st.session_state:  # to not run on switching apps
    st.session_state['filter_pressed'] = False
    del st.session_state['page']

if 'filter_pressed' not in st.session_state:
    st.session_state['filter_pressed'] = False

# Check if the start_date key exists in session_state, if not initialize it
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = date(2023, 1, 23)

# Check if the end_date key exists in session_state, if not initialize it
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = date(2023, 1, 23)

# Check if the rating_filter key exists in session_state, if not initialize it
if 'rating_filter' not in st.session_state:
    st.session_state['rating_filter'] = "All"

# Check if the version key exists in session_state, if not initialize it
if 'version' not in st.session_state:
    st.session_state['version'] = "All"

# Analytics date window wise
st.title('Custom Search by Date Range')
pd.set_option('display.width', 1000)
start_date = st.date_input('Select start date', value=st.session_state['start_date'], min_value=date(
    2023, 1, 1), max_value=datetime.now().date())
end_date = st.date_input('Select end date', value=st.session_state['end_date'], min_value=date(
    2023, 1, 1), max_value=datetime.now().date())
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df['review_date'] = pd.to_datetime(df['review_date'])
dfsearch = df  # to search(full data)
df = df[(df['review_date'] >= start_date) & (df['review_date'] <= end_date)]
rating_options = st.radio("Select Rating Filter", ["All", "4 and below", "5 only"], index=[
                          "All", "4 and below", "5 only"].index(st.session_state['rating_filter']))

# Date Delta
delta = end_date-start_date
day_interval = delta.days+1

# App Version Filter
df_cleaned = df.dropna(subset=['appVersion'])
unique_versions = df_cleaned['appVersion'].drop_duplicates().tolist()
unique_versions.append("All")
unique_versions = df_cleaned['appVersion'].drop_duplicates()
if len(unique_versions) > 0:
    unique_versions = pd.Series(["All"]+unique_versions.tolist())
selected_version = st.selectbox(
    'Select a Version:', unique_versions.tolist(), index=0)

filter_button = st.button('Filter reviews')

# Check if the button is pressed and the dataframe is not empty
if filter_button and not df.empty:
    st.session_state['filter_pressed'] = True

# Save the states of the filters
st.session_state['start_date'] = start_date
st.session_state['end_date'] = end_date
st.session_state['rating_filter'] = rating_options
st.session_state['version'] = selected_version

# Use the filter state to control flow
if st.session_state['filter_pressed'] and not df.empty:
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

    # version wise reviews
    st.title('Review Counts by Version')
    versions = df['appVersion'].value_counts().sort_index()
    fig = go.Figure(data=[go.Bar(x=versions.index, y=versions.values)])
    fig.update_layout(
        title='Review counts by App Version',
        xaxis=dict(title='App Version'),
        yaxis=dict(title='Review Count'),
        barmode='relative'
    )
    st.plotly_chart(
        fig, config={'displayModeBar': False}, use_container_width=True)

    # version release
    dfversion = dfversion.sort_values('appVersion', ascending=False)
    dfversion = dfversion.set_index('appVersion')
    transposed_df = dfversion.T  # Transpose the DataFrame
    st.dataframe(transposed_df)

    # REVIEW TRAFFIC
    df['review_date'] = pd.to_datetime(df['review_date'])
    df.set_index('review_date', inplace=True)
    review_counts = df['review_id'].resample('D').count()
    st.title('Review Distribution Over Time')
    st.subheader('Number of Reviews')
    st.line_chart(review_counts)

    # # Custom Search by Date Range
    # st.title('Custom Search by Date Range')
    # pd.set_option('display.width', 1000)
    # start_date = st.date_input('Select start date')
    # end_date = st.date_input('Select end date')
    # start_date = pd.to_datetime(start_date)
    # end_date = pd.to_datetime(end_date)
    # df_sorted = df.sort_index()  # Sort the DataFrame by the index
    # selected_reviews = df_sorted.loc[start_date:end_date]
    # st.dataframe(selected_reviews)

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

 # Version Review Filtering
    if selected_version != "All":
        df = df[df['appVersion'] == selected_version]
    else:
        df = df

    comments = " ".join(df['review_description'])
    words = word_tokenize(comments)

    def clean_words(new_tokens, custom_stop_words={'angel', 'one', 'please', 'able', 'bank', 'app', 'good', 'nice', 'best'}):
        new_tokens = [t.lower() for t in new_tokens]
        stop_words = set(stopwords.words('english'))
        if custom_stop_words:
            stop_words.update(custom_stop_words)
        new_tokens = [t for t in new_tokens if t not in stop_words]
        new_tokens = [t for t in new_tokens if t.isalpha()]
        lemmatizer = WordNetLemmatizer()
        new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]
        return new_tokens

    cleaned_words = clean_words(words)
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

        # WORDS
        word_freq = word_freq[word_freq['word'] != 'app']
        data = word_freq.head(30).sort_values(by='frequency', ascending=False)
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
        st.dataframe(selected_reviews)


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
                mean_sentiment = sentiment_labels.get(int(mean_ratingrounded))
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
    else:
        st.warning(
            "Please select a different date range to filter the reviews (Not enough data for analysis)")
else:
    st.warning(
        "Please select a different date range to filter the reviews (Not enough data for analysis)")
