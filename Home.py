import streamlit as st


st.set_page_config(
    page_title="App Analytics",
    page_icon="📊",
)

st.title("App Analytics Dashboard")

st.sidebar.success("Select an option above.")

st.info(
    """
    InvestRight and Sky display daily updated reviews, while the remaining apps show reviews limited to the current calendar year.
"""
)
# hide = """
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide, unsafe_allow_html=True)
