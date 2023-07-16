import requests
from time import sleep
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from googletrans import Translator
import numpy as np
import pandas as pd
import time


service = Service(executable_path=r'./chromedriver')
options = Options()
# options.add_argument("--start-maximized")
# options.page_load_strategy = 'normal'
options.add_argument('--headless')
driver = webdriver.Chrome(service=service, options=options)
driver.implicitly_wait(100)

translator = Translator()

channel_name = "yuyuta0702" #put twitch streammer username here

# Navigate to the Twitch stream page
driver.get(f"https://www.twitch.tv/popout/{channel_name}/chat?popout=")

# Check if the element exists
# if elements:
#     print("Element exists on the page")
# else:
#     print("Element does not exist on the page")

# Continuously retrieve chat messages and translate them into english messages

old_msg_list = []

while True:
    chat_messages = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class="chat-line__message"]')))
    if len(chat_messages) != 0:
        for line in chat_messages:
            if line.text not in old_msg_list:
                data = line.text.split(":")
                user = data[0]
                msg = data[1]
                if isinstance(msg, str):
                    translations = translator.translate(msg, dest='en')
                    print(user + ": " + translations.text)
                old_msg_list.append(line.text)
    time.sleep(1)