import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import data as data
from IPython import embed


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=38,  # maxfeats
            embedding_dim=100,
            padding_idx=0,
            max_norm=5.0
        )

        self.conv1d = nn.Conv1d(
            in_channels=100,
            out_channels=256,
            kernel_size=4,
            stride=1
        )

        self.fc = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        emb1 = self.embedding(input_ids).float()
        emb1 = emb1.permute(0, 2, 1)

        conv1 = F.relu(self.conv1d(emb1))
        pool1 = F.max_pool1d(conv1, kernel_size=conv1.shape[2])

        fc1 = torch.cat([pool1.squeeze(dim=2)], dim=1)

        logits = self.fc(self.dropout(fc1))

        return logits


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

    model.eval()
    for inp, lab in testloader:
        out = model(inp)
        y_true.extend(lab.long)
        preds = torch.round(out.squeeze())
        y_pred.extend(preds)

    print(roc_auc_score(y_true, y_pred))


print("fetching data...")
indata = data.get_data()
# Extract data and labels
domains, labels = zip(*indata)
char2ix = {x: idx+1 for idx, x in enumerate(set(''.join(domains)))}
ix2char = {ix: char for char, ix in char2ix.items()}

# Convert characters to int and pad
encoded_domains = [[char2ix[y] for y in x] for x in domains]
encoded_labels = [0 if x == 'benign' else 1 for x in labels]

max_features = len(char2ix) + 1
maxlen = np.max([len(x) for x in encoded_domains])

encoded_labels = np.asarray([
    label for idx, label in enumerate(encoded_labels)
    if len(encoded_domains[idx]) > 1])
encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

assert len(encoded_domains) == len(encoded_labels)

padded_domains = pad_sequences(encoded_domains, maxlen)

X_train, X_test, y_train, y_test = train_test_split(
    padded_domains,
    encoded_labels,
    test_size=0.10
)

trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
trainloader = DataLoader(trainset, batch_size=50, shuffle=True, drop_last=True)

testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
testloader = DataLoader(testset, batch_size=50, shuffle=True, drop_last=True)


lr = 0.001
epochs = 2
clip = 5
step = 0
print_every = 1000

model = CNNModel()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(epochs):
    model.train()
    for inputs, labels in trainloader:
        step += 1
        model.zero_grad()
        y_pred = model(inputs)

        loss = loss_fn(y_pred, labels.long())
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (step % print_every) == 0:
            model.eval()
            eval_losses = []
            for eval_inputs, eval_labels in testloader:
                eval_output = model(eval_inputs)
                eval_loss = loss_fn(eval_output.squeeze(), eval_labels.long())
                eval_losses.append(eval_loss.item())
            print(
                "Epoch: {}/{}".format(epoch+1, epochs),
                "Step: {}".format(step),
                "Training Loss: {:.4f}".format(loss.item()),
                "Eval Loss: {:.4f}".format(np.mean(eval_losses))
            )
    break

embed()
