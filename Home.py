import streamlit as st


st.set_page_config(
    page_title="App Analytics",
    page_icon="📊",
)

st.title("Analytics Dashboard 👋")

st.sidebar.success("Select an option above.")

st.info(
    """
    Investright is updated Real-Time, other apps are
    updated till July 3rd,2023.Data starts from January 1st,2023.
"""
)
# hide = """
#     <style>
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide, unsafe_allow_html=True)
