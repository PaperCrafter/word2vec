import torch
import os
from torch.nn.functional import nll_loss
from model import word2vec
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, config, num_emb, X, y, X_val, y_Val):
        self.config = config
        self.num_emb = num_emb
        self.emb_size = config.emb_size
        self.epoch = config.epoch
        self.lr = config.learning_rate
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_Val

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.losses = []
        self.losses_val = []

        self.build_model()


    def build_model(self):
        self.model = word2vec(self.num_emb, self.emb_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.lr)
        self.model.to(self.device)

    def load_model(self):
        print("Load models from {}...".format(self.model_path))
        path = os.path.join(self.model_path, 'test.pt')

        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print("[*] Model loaded: {}".format(path))

        return self.model

    def train(self):
        for epoch in range(self.epoch):
            print("step %d" % epoch, end="\r")
            self.optimizer.zero_grad()

            #train dataset
            y_pred = self.model.forward(self.X)
            loss = nll_loss(y_pred, self.y).to(self.device)
            self.losses.append(loss.item())
            print("train loss : %f", loss.item())

            #validation dataset
            y_pred_val = self.model.forward(self.X_val)
            loss_val = nll_loss(y_pred_val, self.y_val).to(self.device)
            self.losses_val.append(loss_val.item())
            print("validation loss : %f", loss_val.item())

            loss.backward()
            self.optimizer.step()

        torch.save(self.model.state_dict(), 'test.pt')

    def show_result(self):
        plt.figure(figsize=(8, 8))
        plt.plot(self.losses)
        plt.plot(self.losses_val)
        plt.show()
        print(self.losses[199])
        print(self.losses_val[199])
