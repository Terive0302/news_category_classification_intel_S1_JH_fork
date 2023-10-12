import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_20231012.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_y = encoder.transform(Y)
# fit 트랜스폼을 하면 정보를 새로 가짐, 우리는 정보를 그대로 쓸 것이기 때문에 그냥 트랜스폼
label = encoder.classes_

onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt() # 오늘 받은 뉴스 헤드라인 토큰나이징

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
stopwords = pd.read_csv('./stopwords.csv', index_col=0)

#불용어 제거
for j in range(len(X)): # 뉴스 타이틀 갯수만큼 돎
    words = []
    for i in range(len(X[j])): #
        if len(X[j][i]) > 1: # 1글자 이상 조건문
            if X[j][i] not in list(stopwords['stopword']): # stopwords csv파일에 없는 단어
                words.append(X[j][i]) # 그 단어들만 추가
    X[j] = ' '.join(words)

#불용어 제거한 후 토큰나이징
with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)
# 오늘 크롤링한 뉴스 제목이 더 길 수도 있음 3:40
for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 23: # max크기
        tokened_x[i] = tokened_x[i][:24] # 크기가 max크기보다 크면 max크기만큼 자르기
x_pad = pad_sequences(tokened_x, 23) # max크기

model = load_model('./models/news_category_classification_model_0.7294327020645142.h5')
preds = model.predict(x_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0 # 예측값중에 가장 큰 값의 인덱스를 0으로 초기화, 다시 argmax하면 두번째가 첫번째가 됨
    # 4:11?
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts
print(df.head(30))

# 다시 한 번 정확도 계산
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df)) # 정답 퍼센트 출력
for i in range(len(df)):
    if df['category'][i] not in df['predict'][i]: # 틀린것만
        print(df.iloc[i]) # 출력해보자