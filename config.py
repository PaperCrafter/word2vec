import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default="./book/jekyll_and_hyde.txt", help='path to your dataset')

parser.add_argument('--epoch', type=int, default= 200)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--model_path', default='./', help="path to saved model checkpoints (to continue training)")
parser.add_argument('--window_size', type=int, default= 3)
parser.add_argument('--split', type=float, default= 0.8, help ="The rate at which a dataset is divided into a test dataset and validation dataset")
parser.add_argument('--emb_size', type=int, default= 100, help = "size of vector that consists embbeded word")

#parser.add_argument('--outf', default='checkpoints', help='folder to save model checkpoints')


def get_config():
    return parser.parse_args()