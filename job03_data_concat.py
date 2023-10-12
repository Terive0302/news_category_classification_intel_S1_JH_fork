import pandas as pd
import glob
import datetime

data_path = glob.glob('./crawling_data/*')
print(data_path)

# csv파일 데이터프레임 만들어서 합치기
df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path) # 인덱스가 있는 경우 (경로, index_col = 0) 하면 인덱스 사라짐
    df = pd.concat([df, df_temp])

print(df.head())
print(df['category'].value_counts())
df.info()
df.to_csv('./crawling_data/naver_news_titles_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index = False)
