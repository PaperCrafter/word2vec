from config import get_config
from data_loader import data_loader
from train import Trainer

def main(config):

    data = data_loader(config)
    data_len, X, y, X_val, y_Val = data.preprocess()
    trainer = Trainer(config, data_len, X, y, X_val, y_Val)
    trainer.train()
    trainer.show_result()

if __name__ == "__main__":
    config = get_config()
    main(config)