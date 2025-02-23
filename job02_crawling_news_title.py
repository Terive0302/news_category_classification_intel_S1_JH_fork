from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

options = ChromeOptions()
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
options.add_argument('user-agent=' + user_agent)
options.add_argument("lang=ko_KR")
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument("disable-gpu")
# options.add_argument("--no-sandbox")

# 크롬 드라이버 최신 버전 설정
service = ChromeService(executable_path=ChromeDriverManager().install())

url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1'

# chrome driver
driver = webdriver.Chrome(service=service, options=options)  # <- options로 변경
driver.get(url)

category=['Politics','Economic','Social','Culture','World','IT']
pages=[110,110,110,75,110,72]
df_titles=pd.DataFrame()

for l in range(1, 2):
    section_url='https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}'.format(l)
    #driver.get(url)
    titles=[]
    for k in range(1, pages[l]+1):
        url = section_url+'#&date=%2000:00:00&page={}'.format(k)
        driver.get(url)
        time.sleep(0.5)
        for i in range(1, 5):  # 1부터 5까지
            for j in range(1, 6):  # 1부터 5까지
                try:
                # xpath = '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[{}]/a'.format(i, j, i)
                    title = driver.find_element('xpath', '//*[@id="section_body"]/ul[{}]/li[{}]/dl/dt[2]/a'.format(i, j)).text
                    title = re.compile('[^가-힣]').sub(' ', title)
                    titles.append(title)
                except:
                    print('error {} {} {} {}'.format(l, k, i, j))
        if k % 10 == 0:
            df_section_title = pd.DataFrame(titles, columns = ['titles'])
            df_section_title['category'] = category[l]
            # df_titles=pd.concat([df_titles,df_section_title],ignore_index=True)
            # df_titles.to_csv('./crawling_data/crawling_data_{}_{}.csv'.format(l,k),index=False)
            df_section_title.to_csv('./crawling_data/crawling_data_{}_{}.csv'.format(l ,k) ,index = False)
            titles=[]
    df_section_title = pd.DataFrame(titles, columns = ['titles'])
    df_section_title['category'] = category[l]
    # df_titles = pd.concat([df_titles, df_section_title], ignore_index=True)
    # df_titles.to_csv('./crawling_data/crawling_data_last.csv', index=False)
    df_section_title.to_csv('./crawling_data/crawling_data_last.csv', index=False)

driver.close()
   # titles = []


# print(df_titles.head())
# df_titles.info()
#
# cnt=df_section_title['category'].value_counts()
#
# print(cnt)
#print(title)

# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[2]/dl/dt[2]/a
# //*[@id="section_body"]/ul[1]/li[4]/dl/dt[2]/a

# //*[@id="section_body"]/ul[2]/li[1]/dl/dt[2]/a

# //*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a

# //*[@id="section_body"]/ul[1]/li[1]/dl/dt[2]/a

# //*[@id="section_body"]/ul[4]/li[5]/dl/dt[2]/a