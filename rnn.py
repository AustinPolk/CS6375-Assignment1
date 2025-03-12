import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 2
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer)
        # self.W = nn.Linear(h, 25)
        # self.tanh = nn.Tanh()
        # self.X = nn.Linear(25, 5)
        self.W = nn.Sequential(
            #nn.Linear(h, 5)
            nn.Linear(h, 25),
            #nn.ReLU(),
            nn.Linear(25, 5),
            #nn.ReLU()
            nn.LogSoftmax(dim=0),
        )
        #self.softmax = nn.LogSoftmax(dim=0)
        self.loss = nn.NLLLoss()
        self.do_sum = False

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        output, _ = self.rnn(inputs)
        x = output[:, -1, :]
        #x = self.W(x)
        # x = self.tanh(x)
        # x = self.X(x)

        # if self.do_sum:
        #     x = torch.sum(x, dim=0)
        #     predicted_vector = self.softmax(x)
        # else:
        predicted_vector = self.W(x[-1])

        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    #train_data = train_data[::2]
    #valid_data = valid_data[::2]

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    epoch_by_epoch = {}

    while not stopping_condition:
        epoch_by_epoch[epoch] = {}

        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        start_time = time.time()
        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = np.array([word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words])

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training took {} seconds",format(time.time() - start_time))
        trainning_accuracy = correct/total

        epoch_by_epoch[epoch]['train_acc'] = trainning_accuracy
        epoch_by_epoch[epoch]['train_t'] = time.time() - start_time
        epoch_by_epoch[epoch]['train_avg_loss'] = float(loss_total/loss_count)

        model.eval()

        start_time = time.time()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = np.array([word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words])

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation took {} seconds",format(time.time() - start_time))
        validation_accuracy = correct/total

        epoch_by_epoch[epoch]['val_acc'] = validation_accuracy
        epoch_by_epoch[epoch]['val_t'] = time.time() - start_time

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy and epoch >= args.epochs:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

    torch.save(model, 'rnn_model.pt')
    with open(f'rnn_last_model_results_h{args.hidden_dim}_e{epoch-1}_l{model.numOfLayer}.pkl', 'wb+') as f:
        pickle.dump(epoch_by_epoch, f)

    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
