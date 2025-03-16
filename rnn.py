import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
from ffnn import FFNN

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer)
        self.softmax = nn.LogSoftmax(dim=0)
        self.W = FFNN(h, 32)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        output, _ = self.rnn(inputs)
        x = output[:, -1, :]
        return self.W(x[-1])

def load_data(train_data, val_data, val2_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    if val2_data:
        with open(val2_data) as valid2_f:
            validation2 = json.load(valid2_f)

    tra = []
    val = []
    val2 = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    if val2_data:
        for elt in validation2:
            val2.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, val2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--val2_data", default = None, help = "path to extra validation data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, valid2_data = load_data(args.train_data, args.val_data, args.val2_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    
    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    optimizer = optim.SGD(model.parameters(), lr=0.075, momentum=0)
    #optimizer = optim.Adam(model.parameters(), lr=0.03)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

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
        minibatch_size = 8
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

        if args.val2_data:
            model.eval()

            start_time = time.time()
            correct = 0
            total = 0
            random.shuffle(valid2_data)
            print("Extra validation started for epoch {}".format(epoch + 1))
            valid2_data = valid2_data

            for input_words, gold_label in tqdm(valid2_data):
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
            print("Extra validation completed for epoch {}".format(epoch + 1))
            print("Extra validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Extra validation took {} seconds",format(time.time() - start_time))
            validation2_accuracy = correct/total

            epoch_by_epoch[epoch]['val2_acc'] = validation2_accuracy
            epoch_by_epoch[epoch]['val2_t'] = time.time() - start_time

        if epoch >= args.epochs:
            stopping_condition=True
            print("Training done!")

        epoch += 1

    with open(f'rnn_training_results_h{args.hidden_dim}.pkl', 'wb+') as f:
        pickle.dump(epoch_by_epoch, f)