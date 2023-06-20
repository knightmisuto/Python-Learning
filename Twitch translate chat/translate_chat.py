import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from googletrans import Translator
import time

options = webdriver.ChromeOptions()
options.add_argument("--headless")

browser = webdriver.Chrome("chromedriver.exe", chrome_options=options)
translator = Translator()


channel_name = "" #put twitch streammer username here
url = f"https://www.twitch.tv/popout/{channel_name}/chat?popout="


browser.get(url)

old_msg_list = []

while True:
    soup = bs(browser.page_source, "lxml")
    data_big = soup.find_all("div", {"class": "chat-line__message"}) 
    for line in data_big:
        if line.text not in old_msg_list:
            data = line.text.split(":")
            user = data[0]
            msg = data[1]
            translations = translator.translate(msg, dest='en')
            print(user + ": " + translations.text)
            old_msg_list.append(line.text)
    time.sleep(5)