import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

DATA_PATH = "/Users/gauthamys/Desktop/UIC_CS483_Final_Project/data/lyrics"

df = pd.concat([pd.read_csv(f'{DATA_PATH}/{file}') for file in os.listdir(DATA_PATH)])


max_words = 10000  # Max number of words to keep in the tokenizer
max_len = 100  # Max length for each lyric sequence (truncated/padded)

# Initialize Tokenizer and fit on lyrics
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['lyrics'])

# Convert lyrics to sequences of integers
sequences = tokenizer.texts_to_sequences(df['lyrics'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Model Architecture
embedding_dim = 128  # Embedding size for each token
lstm_units = 64  # Number of LSTM units

model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(lstm_units, return_sequences=False),
    Dense(embedding_dim, activation='relu')  # Final dense layer to produce the vector representation
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Generate the vectors for each lyric by passing padded sequences through the model
lyric_vectors = model.predict(padded_sequences)

# Convert the array to a DataFrame if needed, or add as a column in df
lyric_vectors_df = pd.DataFrame(lyric_vectors)
df = pd.concat([df, lyric_vectors_df], axis=1)

# Display the result
print(df)