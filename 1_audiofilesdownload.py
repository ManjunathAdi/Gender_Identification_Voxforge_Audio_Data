# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:16:27 2017

@author: manjunath.a
"""


from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd 


driver = webdriver.Chrome('C:\\Users\\manjunath.a\\Downloads\\chromedriver_win32\\chromedriver')
 
driver.get("http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/")

wait = WebDriverWait(driver, 600)



data = pd.read_csv("C:\\Users\\manjunath.a\\Downloads\\Audiofiles.csv")
audiofiles = data['Audiofiles']


for i in audiofiles:
    Menu = '//a[\''+i+'\')]'
    Menu_button = wait.until(EC.presence_of_element_located((
        By.XPATH, Menu)))
    Menu_button.click()
    time.sleep(2)
    
      
    