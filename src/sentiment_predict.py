import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import SimpleRNN, Dense, Embedding

data = pd.read_csv('datasets/swiggy.csv')
print('Columns in datasets:')
print(data.columns.to_list())

data["Review"] = data["Review"].str.lower().replace(
    r'[^a-z0-9\s]', '', regex=True)

data['sentiment'] = data['Avg Rating'].apply(lambda x: 1 if x > 3.5 else 0)

max_features = 5000
max_length = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['Review'])
x = pad_sequences(tokenizer.texts_to_sequences(
    data["Review"]), maxlen=max_length)
y = data['sentiment'].values

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

model = Sequential([
    Embedding(input_dim=max_features, output_dim=16, input_length=max_length),
    SimpleRNN(64, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                    validation_data=(x_val, y_val), verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {score[1]:.2f}')


def predict_sentiment(review_text: str):
    text = review_text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)
    prediction = model.predict(padded)[0][0]
    return f"{'Positive' if prediction >= 0.5 else "Negative"} (Probability: {prediction:.2f})"


sample_review = "The food was great!"
print(f"Review: {sample_review}")
print(f"Sentiment: {predict_sentiment(sample_review)}")
