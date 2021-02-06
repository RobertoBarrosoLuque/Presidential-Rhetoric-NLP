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
    chromedriver = "C:/Users/robal/Downloads/chromedriver_win32chromedriver/chromedriver.exe" # path to the chromedriver executable
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(ChromeDriverManager().install())
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.get('https://millercenter.org/the-presidency/presidential-speeches')

    for i in range(400):
        #Alternatively, document.body.scrollHeight
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight*10);")
    #Wait for page to load
    time.sleep(3)

    # Get all relevant partial links
    speeches = []
    soup1 = BeautifulSoup(driver.page_source, 'html.parser')
    rows1 = soup1.find_all('div', class_ = 'views-row')
    for x in rows1:
        endings = x.find('a')['href'][38:]
        speeches.append(endings)

    base_url = 'https://millercenter.org/the-presidency/presidential-speeches/'
    presidents = {}

    len(speeches)
    speeches
    for i in speeches:
        full_url = base_url + i
        print(full_url)

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
        df_uva = df_dict.T
        df_uva.columns = ['President', 'Date', 'Speech']
        df_uva.shape
