import pandas as pd
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import os
import time
from webdriver_manager.chrome import ChromeDriverManager

def get_president_speech():
    """
    Function to process and download presidential speeches corpus.
    This function will download and save a .xlsx file with presidential speeches
    Based on the following github project:
    https://github.com/nicolesemerano/Metis-Project-4-Presidential-Speeches-NLP
    """
    chromedriver = "C:/Users/robal/Downloads/chromedriver_win32chromedriver/chromedriver.exe" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(ChromeDriverManager().install())
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.get('https://millercenter.org/the-presidency/presidential-speeches')

    for i in range(4000):
        #Alternatively, document.body.scrollHeight
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight*10);")

    #Wait for page to load
    time.sleep(3)

    # Get all relevant partial links
    speeches = []
    soup1 = BeautifulSoup(driver.page_source, 'html.parser')
    rows1 = soup1.find_all('div', class_ = 'views-row')

    speeches = [x.find('a')['href'][38:] for x in rows1]

    base_url = 'https://millercenter.org/the-presidency/presidential-speeches/'
    presidents = {}
    print("Scrolled down to get a total of {} speeches".format(len(speeches)))

    speeches
    for i in speeches:
        full_url = base_url + i
        response_all = requests.get(full_url)
        soup_all = BeautifulSoup(response_all.text,'html.parser')
        try:
            name = soup_all.find(class_ = 'president-name').text
            date = soup_all.find(class_ = 'episode-date').text
            transcript = soup_all.find(class_="view-transcript").text[10:]
        except AttributeError:
            print("Attribute error failed to download:")
            print(full_url)
            continue

        presidents[i] = [name] + [date] + [transcript]

        #Convert into pandas data frame
        df_dict = pd.DataFrame(presidents)
        final_df = df_dict.T
        final_df.columns = ['President', 'Date', 'Speech']
        final_df["Speech"] = final_df.Speech.apply(lambda row:
                                                   row.replace("anscript",
                                                   "",1).replace("Transcript",
                                                   ""), 1)
        final_df.to_excel("presidential_speeches.xlsx")


def main():
    get_president_speech()
