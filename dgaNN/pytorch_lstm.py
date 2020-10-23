import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import dga_classifier.data as data
from IPython import embed


class LSTMModel(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, n_layers, batch_size):
        super(LSTMModel, self).__init__()
        
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(feat_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden):
        embedded_feats = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded_feats, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        fc_out = self.fc(lstm_out)
        sigmoid_out = self.sigmoid(fc_out)
        sigmoid_out = sigmoid_out.view(self.batch_size, -1)
        sigmoid_last = sigmoid_out[:,-1]

        return sigmoid_last, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
             weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        return h

def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0]*(maxlen-len(domain))+domain)
    return np.asarray(domains)

def evaluate(model, testloader, batch_size):
    y_pred = []
    y_true = []

    h = model.init_hidden(batch_size)
    model.eval()
    for inp, lab in testloader:
        h = tuple([each.data for each in h])
        out, h = model(inp, h)
        y_true.extend(lab)
        preds = torch.round(out.squeeze())
        y_pred.extend(preds)

    print(roc_auc_score(y_true, y_pred))

def run():
    print("fetching data...")
    indata = data.get_data()
    # Extract data and labels
    domains, labels = zip(*indata)
    char2ix = {x:idx+1 for idx, x in enumerate(set(''.join(domains)))}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 'benign' else 1 for x in labels]

    max_features = len(char2ix) + 1
    maxlen = np.max([len(x) for x in encoded_domains])

    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]
    
    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)

    X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10)

    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
    trainloader = DataLoader(trainset, batch_size=50, shuffle=True, drop_last=True)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
    testloader = DataLoader(testset, batch_size=50, shuffle=True, drop_last=True)

    embed_size = 128
    hidden_size = 128
    n_layers = 2
    batch_size = 50
    
    lr = 0.001
    epochs = 2
    clip = 5
    step = 0
    print_every = 1000
    
    model = LSTMModel(max_features, embed_size, hidden_size, n_layers, batch_size)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        h = model.init_hidden(batch_size)
        for inputs, labels in trainloader:
            step +=1
            h = tuple([each.data for each in h])

            model.zero_grad()
            output, h = model(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # running_loss += loss.item()
            if (step % print_every) == 0:
                val_h = model.init_hidden(batch_size)
                model.eval()
                eval_losses= []
                for eval_inputs, eval_labels in testloader:
                    val_h = tuple([each.data for each in val_h])
                    eval_output, val_h = model(eval_inputs, val_h)
                    eval_loss = criterion(eval_output.squeeze(), eval_labels.float())
                    eval_losses.append(eval_loss.item())

                print(
                    "Epoch: {}/{}".format(epoch+1, epochs),
                    "Step: {}".format(step),
                    "Training Loss: {:.4f}".format(loss.item()),
                    "Eval Loss: {:.4f}".format(np.mean(eval_losses))
                )
                model.train()

    embed()