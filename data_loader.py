import re
import json
import numpy as np
import torch

class data_loader():
    def __init__(self, config):
        self.config = config
        self.path = config.data_path
        self.wSize = config.window_size
        self.split = config.split

    def load_txt(self):
        # Open and read text file
        text_file = open(self.path, encoding='utf8')
        new = text_file.read()
        return new

    # Remove punctuation and separate words
    def clean_words(self, file):
        clean = re.sub("[^a-zA-Z']", " ", file)
        clean = clean.lower()
        words = clean.split()
        return words


    def create_pairs(self, words, dictionary, window_size):
        pairs = []
        for index in range(window_size, len(words) - window_size):
            for i in range(1,window_size+1):
                pairs.append([dictionary[words[index]],dictionary[words[index-i]]])
                pairs.append([dictionary[words[index]],dictionary[words[index+i]]])
        return(pairs)

    def preprocess(self):
        txt = self.load_txt()
        words = self.clean_words(txt)
        print(len(words))

        words = words[:10000]
        # Unique words
        vocab = list(set(words))

        # Words paired with their index in vocab so we can use for word embeddings
        vocab_dict = {word: i for i, word in enumerate(vocab)}

        with open("vocab_dict.txt", "w") as f:
            f.write(json.dumps(vocab_dict))

        pears = np.array(self.create_pairs(words, vocab_dict, self.wSize))
        print("len pairs: " + str(len(pears)))

        # Divide pairs into training and validation
        a = int(np.floor(len(pears) * self.split))

        # Create training and validation data

        data_X_train = pears[:a, 0]
        data_y_train = pears[:a, 1]

        data_X_val = pears[a:, 0]
        data_y_val = pears[a:, 1]

        X = torch.from_numpy(data_X_train).long()
        y = torch.from_numpy(data_y_train).long()

        X_val = torch.from_numpy(data_X_val).long()
        y_val = torch.from_numpy(data_y_val).long()

        return len(vocab), X, y, X_val, y_val


