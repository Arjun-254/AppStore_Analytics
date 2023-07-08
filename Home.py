import streamlit as st


st.set_page_config(
    page_title="App Analytics",
    page_icon="📊",
)

st.title("App Analytics Dashboard")

st.sidebar.success("Select an option above.")

st.info(
    """
    InvestRight and Sky display all daily updated reviews, while the remaining apps show the last 5000 reviews.
"""
)
# hide = """
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide, unsafe_allow_html=True)
