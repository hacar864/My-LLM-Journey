#%% Importing the libs
import re
import numpy as np

#%% Exercise 1: Make encoder and decoder functions

text = ["All that we are is the result of what we have thought",
        "To be or not to be that is the question",
        "Be yourself everyone else is already taken"]

all_words = re.split( "\s", " ".join(text).lower() )

vocab = sorted(set(all_words))

print(f"There are {len(all_words)} words in the text and It contains {len(vocab)} unique words.")
print(all_words)

#%% Defining the dictionaries and functions

word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

idx2word = {}
for i, word in enumerate(vocab):
    idx2word[i] = word

def encoder(x):
    numbers = []
    for i in x:
        numbers.append(word2idx[i])

    return numbers

def decoder(x):
    words = []
    for i in x:
        words.append(idx2word[i])

    return words

# Tests
tokens = encoder(all_words)
print(tokens)

words = decoder(tokens)
print(words)

# Inverse functions
print(decoder(encoder(all_words))) # This should return the input.








