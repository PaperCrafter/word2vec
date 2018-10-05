from scipy.spatial.distance import cdist
import torch
from model import word2vec
import numpy as np
from config import get_config
from data_loader import data_loader
import json


def get_distance(word1, word2, emb, dist_type):
    index1 = vocab_dict[word1]
    index2 = vocab_dict[word2]

    print(index1)
    print(index2)

    emb1 = [emb[index1]]
    emb2 = [emb[index2]]

    distance = cdist(emb1, emb2, dist_type)

    return distance

config = get_config()
data = data_loader(config)
num_emb, _, _, _, _ = data.preprocess()

model = word2vec(num_emb, config.emb_size)
model.load_state_dict(torch.load("./test.pt"))

with open('./vocab_dict.txt','r') as f:
    vocab_dict = json.loads(f.read())
    print(type(vocab_dict))
print(vocab_dict)


params = []
filename = "parameters.txt"

# write embedded words
with open("parameters.txt", "w") as f:
    for param in model.parameters():
        for thing in param:
            f.write(str(thing.data))
        params.append(param.data)

embeddings = params[1].numpy()
# print(len(embeddings))
np.transpose(embeddings)

# print(len(embeddings))



print(get_distance('person', 'girl', embeddings, 'cosine'))

distance1 = get_distance('girl', 'party', embeddings,  'euclidean')
distance2 = get_distance('embarrassed', 'footstep', embeddings,  'euclidean')
distance3 = get_distance('habit', 'spirits', embeddings,  'euclidean')
distance4 = get_distance('extraordinary', 'circumstance', embeddings,  'euclidean')
print(distance1)
print(distance2)
print(distance3)
print(distance4)