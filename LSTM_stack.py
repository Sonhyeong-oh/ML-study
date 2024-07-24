import pandas as pd
import tensorflow as tf

# LSTM 모델 생성
def create_lstm_model(seq_length):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(seq_length.size-1, 1), return_sequences=True))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

stack = pd.read_excel("C:/Users/daily/Desktop/data.xlsx")
df = pd.DataFrame(stack)
df = df.dropna(subset='future_log_returns')
input = df.drop('future_log_returns', axis = 1)
target = df['future_log_returns']

# LSTM 모델 생성
model = create_lstm_model(input)

# 모델 컴파일
model.compile(loss='mse', optimizer='adam')

# 모델 훈련
model.fit(input, target, epochs=10, batch_size=32)

# 새로운 시퀀스 생성 및 예측
new_input = input[:-1]
predicted_output = model.predict(new_input)
print("입력 시퀀스:", input[:-1])
print("예측된 다음 값:", predicted_output.flatten())