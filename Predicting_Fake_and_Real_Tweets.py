# 1. Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
import time
import os
import re
import string

#1.a Defines
SPLIT_SIZE = 0.8
MAX_LENGTH = 20
EMBEDDING_DIM = 32

#1.b Helper functions
def remove_url(text):
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub( r"", text)

def remove_punct(text):
    translator = str.maketrans("","",string.punctuation)
    return text.translate(translator)

def counter_words(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

def split_data(input_data, length, split_size):
    train_size = int(split_size * length)

    train_data = input_data[:train_size]
    test_data = input_data[train_size:]

    train_sentences = train_data.text.to_numpy()
    train_labels = train_data.target.to_numpy()

    test_sentences = test_data.text.to_numpy()
    test_labels = test_data.target.to_numpy()

    return train_sentences, train_labels, test_sentences, test_labels

def decode(reverse_word_idx,sequence):
    return " ".join([reverse_word_idx.get(idx, "?") for idx in sequence])

# 2. Load data into environemnt
source_path = os.path.join('D:', 'Data', 'twitter_train.csv')
if os.path.exists(source_path):
    print('File exists')
else:
    print('No file found')
print('The file path is:', source_path)

data = pd.read_csv(source_path)

# 3. visualize the data
data_row, data_col = data.shape
print(data.shape)
print(data.head())

disaster_total = (data.target == 1).sum()
not_disaster_total = (data.target == 0).sum()

#3.a is the data balanced
print(disaster_total) #total count of disaster
print(not_disaster_total) #total count of no disaster

#plt.bar(['total count of disaster', 'total count of no disaster'], [disaster_total, not_disaster_total])
#plt.show()

# 4. Process the data
print(string.punctuation)
counter = counter_words(data.text)
print(len(counter))
#print(counter)
print(counter.most_common(5))

number_of_unique_word = len(counter)

train_sentences, train_labels, test_sentences, test_labels = split_data(data, data_row, SPLIT_SIZE)

# 5. Validate the data shape
print(train_sentences.shape)
print(test_sentences.shape)

# 6. Tokenize the data
tokenizer = Tokenizer(num_words=number_of_unique_word)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)

print(train_sequences[10:15])
print(train_sentences[10:15])

train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

print("traiing data padded shape is", train_padded.shape)
print("test data padded shape is", test_padded.shape)

#check to see if padding worked
print("testing the paddding")
print(train_padded[10])
print(train_sentences[10])
print(train_sequences[10])

#reverse the indicies
reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])
print("testing the decode")
decoded_text = decode(reverse_word_index, train_sequences[10])
print("The sequence is: ", train_sequences[10])
print("The decoded text is: ", decoded_text)

# 7. Model,compile, fit
model = Sequential()
model.add(Embedding(number_of_unique_word, EMBEDDING_DIM, input_length=MAX_LENGTH))
model.add(Bidirectional(tf.keras.layers.LSTM(150)))
#model.add(tf.keras.layers.LSTM(64, dropout=0.1))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# Compile the model
model.compile(loss=losses.BinaryCrossentropy(from_logits=False),
               optimizer=optimizers.Adam(lr=0.001),
               metrics=['accuracy'])

history = model.fit(train_padded, train_labels, epochs=20, validation_data=(test_padded, test_labels))

predictions = model.predict(train_padded)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print("The training sentences are:", train_sentences[10:20])
print("The training lables are:", train_labels[10:20])
print("the predicitons are: ", predictions[10:20])
