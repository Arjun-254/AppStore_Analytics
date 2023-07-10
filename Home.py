import streamlit as st


st.set_page_config(
    page_title="App Analytics",
    page_icon="📊",
)

st.title("Apple App Store Analytics Dashboard")

st.sidebar.success("Select an option above.")

st.info(
    """
    InvestRight and Sky display all daily updated reviews, while the remaining apps show the latest 5000 reviews
"""
)
# hide = """
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide, unsafe_allow_html=True)
