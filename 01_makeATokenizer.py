#%% Importing the libs
import re
import numpy as np
import matplotlib.pyplot as plt

#%% Exercise 1: Defining the dictionaries

text = ["All that we are is the result of what we have thought",
        "To be or not to be that is the question",
        "Be yourself everyone else is already taken"]

all_words = re.split( "\s", " ".join(text).lower() )

vocab = sorted(set(all_words))

print(f"There are {len(all_words)} words in the text and It contains {len(vocab)} unique words.")
print(all_words)

word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

idx2word = {}
for i, word in enumerate(vocab):
    idx2word[i] = word

#%% Exercise 2: Defining encoder and decoder functions

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

new_text = "We already are the result of what everyone else already thought"
all_newText_words = re.split("\s", new_text.lower())

encoded_newText = encoder(all_newText_words)
decoded_newText = decoder(encoded_newText)

print("New Text:")
print(f"\t{new_text}")

print("\nToken IDs:")
print(f"\t{encoded_newText}")

print("\nDecoded Text:")
print(f"\t{decoded_newText}")

#%% Exercise 3: Visualize the tokenized integers (I got some help in here)

tokens = encoder(all_words)

_, ax = plt.subplots(1, figsize=(12,5))

ax.plot(tokens, "bs", markersize=10)
ax.set(xlabel="Word Index", yticks=range(len(vocab)))
ax.grid(linestyle="--", axis="y")

ax2 = ax.twinx()
ax2.plot(tokens, alpha=0)
ax2.set(yticks=range(len(vocab)), yticklabels=vocab)

plt.show()

#%% Exercise 4: Explore context surrounding target tokens (I got some help in here)

target_word = "to"
target_token = word2idx[target_word]

target_Locs = np.where(np.array(all_words)==target_word)[0]
print(f"'{target_word}' appears at indices {target_Locs}\n")

for t in target_Locs:
    print(tokens[t-1:t+2])
    print(" ".join(all_words[t-1:t+2]), "\n")













