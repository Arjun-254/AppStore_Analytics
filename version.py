from bs4 import BeautifulSoup
import requests
import pandas as pd
import streamlit as st

# Function to scrape version info


def scrape_version_info(link):
    html_text = requests.get(link).text
    soup = BeautifulSoup(html_text, 'html.parser')
    latest_update = soup.find(
        "section", class_='l-content-width section section--bordered whats-new')

    version = latest_update.find(
        'p', class_='l-column small-6 medium-12 whats-new__latest__version').text
    date = latest_update.find('time').text
    div = latest_update.find(
        'div', class_='we-truncate we-truncate--multi-line we-truncate--interactive')
    notes = div.find('p').text
    notes = notes.replace('-', '->')
    return version, date, notes


def version_info(link, app_number):
    st.title(f"Release Information")
    st.write("Here is the release information:")
    version, date, notes = scrape_version_info(link)

    # Load or create the DataFrame using the app number
    if f"version_df_{app_number}" not in st.session_state:
        st.session_state[f"version_df_{app_number}"] = pd.DataFrame(
            columns=["Version", "Release Date", "Notes"])

    # Check if the version is not already present in the stored DataFrame, if not append the new version data to the stored DataFrame
    if version not in st.session_state[f"version_df_{app_number}"]["Version"].values:
        new_row = {"Version": version, "Release Date": date, "Notes": notes}
        new_df = pd.DataFrame([new_row])
        st.session_state[f"version_df_{app_number}"] = pd.concat(
            [st.session_state[f"version_df_{app_number}"], new_df], ignore_index=True)

    st.dataframe(st.session_state[f"version_df_{app_number}"],
                 hide_index=True, use_container_width=True)


# def version_info(link):
#     st.title("Release Information")
#     st.write("Here is the release information:")
#     html_text = requests.get(link).text
#     soup = BeautifulSoup(html_text, 'html.parser')
#     latest_update = soup.find(
#         "section", class_='l-content-width section section--bordered whats-new')

#     version = latest_update.find(
#         'p', class_='l-column small-6 medium-12 whats-new__latest__version').text
#     date = latest_update.find('time').text
#     div = latest_update.find(
#         'div', class_='we-truncate we-truncate--multi-line we-truncate--interactive')
#     notes = div.find('p').text
#     notes = notes.replace('-', '->')

#     data = {
#         "Version": [version],
#         "Release Date": [date],
#         "Notes": [notes]
#     }

#     # Create a DataFrame from the dictionary
#     df = pd.DataFrame(data)
#     # Display the cached DataFrame
#     st.dataframe(df, hide_index=True, use_container_width=True)
