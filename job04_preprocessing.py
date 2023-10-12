import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./crawling_data/naver_news_titles_20231012.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

encoder = LabelEncoder() # encoder가 클래스에 대한 정보를 가지고 있음
labeled_y = encoder.fit_transform(Y) # fit 트랜스폼하면 0 1 2 3 4 5 로 바뀜 12:36, 2:02
# print(labeled_y[:3])
label = encoder.classes_
# print(label)

with open('./models/encoder.pickle', 'wb') as f: # 텍스트파일이 아니기 때문에 'w'하면 안됨, 'wb'로 바이너리로 저장
    pickle.dump(encoder, f)                      # 읽을 때도 똑같이 바이너리 파일이기 때문에 'rb'


onehot_y = to_categorical(labeled_y)
# print(onehot_y)

okt = Okt()

for i in range(len(X)):
    # okt_morph_x = okt.morphs(X[0], stem = True)
    X[i] = okt.morphs(X[i], stem = True)
    # stem = True를 주면 원형으로 바꿔줌 (하다)
    # stem = True를 안주면 동사 변형(하며 하는 하니 등)으로 되므로 학습이 잘 안됨
    # 데이터가 많으면 상관없지만 우리는 데이터가 적기 때문에 원형으로 바꿔서 학습이 잘 되도록 해야 함 11:35 40
# print(X[0])

# 전처리 / stopword, 불용어 제거
stopwords = pd.read_csv('./stopwords.csv', index_col = 0)
for j in range(len(X)): # 뉴스 타이틀 갯수만큼 돎
    words = []
    for i in range(len(X[j])): #
        if len(X[j][i]) > 1: # 1글자 이상 조건문
            if X[j][i] not in list(stopwords['stopword']): # stopwords csv파일에 없는 단어
                words.append(X[j][i]) # 그 단어들만 추가
    X[j] = ' '.join(words)
# print(X[0])

# 라벨링 # 형태소 하나 하나에 라벨을 부여해서 라벨로 바꿔주는 작업
token = Tokenizer()
token.fit_on_texts(X) # 2:03
tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print(tokened_x[0:3])
print(wordsize) # 12:26 32

# 저장하기 12:33
with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f) # 토큰을 f에다 저장

# 2:07 2:11
max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])
print(max)

# 모든 문장의 길이를 max로 맞추기 (23)
x_pad = pad_sequences(tokened_x, max) # 문장을 주고, 길이를 주면 모자란 갯수만큼 0을 채워줌
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)