import torch
from torch.nn.functional import binary_cross_entropy,softmax, nll_loss, log_softmax

class word2vec(torch.nn.Module):
    def __init__(self, num_emb, emb_size):
        super(word2vec, self).__init__()
        #         Create initial word embeddings
        self.emb1 = torch.nn.Embedding(num_emb, emb_size)

        #     Weights, which will be updated with each cycle and represent the final word vectors
        self.W = torch.nn.Parameter(torch.rand(emb_size, num_emb))

    def forward(self, x):
        #     Create word embeddings
        x = self.emb1(x)
        #     Multiply word embeddings with weights
        x = torch.matmul(x, self.W)

        return log_softmax(x, dim=1)

