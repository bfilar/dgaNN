from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

import data as data
from IPython import embed


class CNNModel(nn.Module):
    def __init__(self, embed_size):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=embed_size, embedding_dim=100, padding_idx=0
        )

        self.conv1d = nn.Conv1d(
            in_channels=100, out_channels=256, kernel_size=4, stride=1
        )
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        emb1 = self.embedding(input_ids).float()
        emb1 = emb1.permute(0, 2, 1)

        conv1 = F.relu(self.conv1d(emb1))
        pool1 = F.max_pool1d(conv1, kernel_size=conv1.shape[2])

        fc = torch.cat([pool1.squeeze(dim=2)], dim=1)
        fc1 = self.fc1(self.dropout(fc))
        dropout = self.dropout(fc1)
        fc_out = self.fc2(dropout)
        sigmoid_out = self.sigmoid(fc_out)
        return sigmoid_out


def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0] * (maxlen - len(domain)) + domain)
    return np.asarray(domains)


print("fetching data...")
indata = data.get_data()
# Extract data and labels
domains, labels = zip(*indata)
char2ix = {x: idx + 1 for idx, x in enumerate(set("".join(domains)))}
ix2char = {ix: char for char, ix in char2ix.items()}

# Convert characters to int and pad
encoded_domains = [[char2ix[y] for y in x] for x in domains]
encoded_labels = [0 if x == "benign" else 1 for x in labels]

max_features = len(char2ix) + 1
maxlen = np.max([len(x) for x in encoded_domains])

encoded_labels = np.asarray(
    [label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1]
)
encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

assert len(encoded_domains) == len(encoded_labels)

padded_domains = pad_sequences(encoded_domains, maxlen)

X_train, X_test, y_train, y_test = train_test_split(
    padded_domains, encoded_labels, test_size=0.10
)

trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
trainloader = DataLoader(trainset, batch_size=50, shuffle=True, drop_last=True)

testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
testloader = DataLoader(testset, batch_size=50, shuffle=True, drop_last=True)

lr = 0.001
epochs = 2
step = 0
print_every = 1000

model = CNNModel(embed_size=max_features)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

for epoch in range(epochs):
    model.train()
    for inputs, labels in trainloader:
        step += 1
        model.zero_grad()
        y_pred = model(inputs)

        loss = loss_fn(y_pred.view(-1), labels)
        loss.backward()
        optimizer.step()

        if (step % print_every) == 0:
            model.eval()
            metrics = defaultdict(list)
            for eval_inputs, eval_labels in testloader:
                eval_output = model(eval_inputs)
                eval_loss = loss_fn(eval_output.squeeze(), eval_labels)
                metrics["eval_losses"].append(eval_loss.item())
                ypred = torch.round(eval_output.squeeze()).detach().numpy()
                metrics["ypred"].extend(list(ypred))
                metrics["ytrue"].extend(list(eval_labels.numpy()))

            eval_auc = roc_auc_score(metrics["ypred"], metrics["ytrue"])
            print(
                "Epoch: {}/{}".format(epoch + 1, epochs),
                "Step: {}".format(step),
                "Training Loss: {:.4f}".format(loss.item()),
                "Eval Loss: {:.4f}".format(np.mean(metrics["eval_losses"])),
                "Eval AUC: {:.4f}".format(eval_auc),
            )

print(classification_report(metrics["ypred"], metrics["ytrue"]))

embed()
