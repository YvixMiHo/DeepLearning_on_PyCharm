import re
import string
from collections import Counter

def remove_url(text):
    """ removes URL from text """
    url = re.compile(r"https?://\S+|www\.\S+")
    return url.sub( r"", text)

def remove_punct(text):
    """ removes punctuation from text """
    translator = str.maketrans("","",string.punctuation)
    return text.translate(translator)

def counter_words(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

def split_data(input_data, length, split_size):
    """ splits data and returns training, test sentences and labels """
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