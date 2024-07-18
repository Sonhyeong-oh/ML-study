import numpy as np
import tensorflow as tf

def generate_sequence(length):
    freq = np.random.uniform(0.1, 1.0)
    x = np.sin(np.arange(0, length) * freq)
    return x

def generate_dataset(seq_length, n_samples):
    X = []
    Y = []
    for _ in range(n_samples):
        seq = generate_sequence(seq_length)
        X.append(seq[:-1])  # 입력 시퀀스
        Y.append(seq[1:])   # 출력 시퀀스
    X = np.array(X)
    X = np.reshape(X, [n_samples, seq_length-1, 1])
    Y = np.array(Y)
    Y = np.reshape(Y, [n_samples, seq_length-1, 1])
    return X, Y

# LSTM 모델 생성
def create_lstm_model(seq_length):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(20, input_shape=(seq_length-1, 1), return_sequences=True))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

# 데이터셋 생성
seq_length = 10
n_samples = 1000
X, Y = generate_dataset(seq_length, n_samples)

# LSTM 모델 생성
model = create_lstm_model(seq_length)

# 모델 컴파일
model.compile(loss='mse', optimizer='adam')

# 모델 훈련
model.fit(X, Y, epochs=10, batch_size=32)

# 새로운 시퀀스 생성 및 예측
new_seq = generate_sequence(seq_length)
new_input = new_seq[:-1]
new_input = np.reshape(new_input, [1, seq_length-1, 1])
predicted_output = model.predict(new_input)
print("입력 시퀀스:", new_seq[:-1])
print("예측된 다음 값:", predicted_output.flatten())
