import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data/news_data_max_23_wordsize_14301.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(14301, 300, input_length=23)) # 자연어 학습할 때 중요한 레이어
# 단어들을 단어 갯수만큼의 차원을 가지는 공간상의 배치, 공간상의 벡터값으로 바꿔줌 2:40 50
# 의미공간상의 벡터화, Embedding layer가 해줌
# 300은 차원 축소 3:05
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# 문장은 한줄 1차원이니까 1D를 씀, 필터 32개, 3:11 12
model.add(MaxPooling1D(pool_size=1)) # 텐서플로우 버전때문에 맥스풀이 아니라 맥스풀링이 됨
# 풀 사이즈 1이므로 아무 일도 일어나지 않음, 빼도 되는 레이어지만 Conv레이어를 써주면 같이 써주는게 좋음
model.add(LSTM(128, activation='tanh', return_sequences=True))
# 순서에 따른 학습을 위해 LSTM, return_sequences=True 하나 들어갈 때마다 저장, 이게 없으면 맨 마지막 출력만 내놓음 3:14
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
# 다음에 LSTM이 있으니 리턴 시퀀스 True
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
# 다음 LSTM없으니 리턴 시퀀스 없음
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax')) # 카테고리 갯수만큼 레이어, 액'펑은 소프트맥스
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()