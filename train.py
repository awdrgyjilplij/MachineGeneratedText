from pathlib import Path
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

dataPath = Path('data')
savePath = Path('model.pkl')
trainPath = Path(dataPath, 'train.txt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# build the dataset
class textData(Dataset):
    def __init__(self, path, transform=None):
        self.lines = []
        self.labels = []
        self.transform = transform
        with open(path, encoding='utf-8') as in_file:
            for line in in_file:
                line = line.strip()
                self.lines.append(line[:-1])
                self.labels.append(int(line[-1]))
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        line = self.lines[index]
        label = self.labels[index]
        sample = line, label
        if self.transform:
            sample = self.transform(sample)
        return sample

# calculate the accuracy
def calc_acc(outputs, labels):
    output_labels = []
    for item in outputs:
        if item[0] > item[1]:
            output_labels.append(0)
        else:
            output_labels.append(1)
    acc = 0
    n = len(labels)
    for i in range(n):
        if output_labels[i] == labels[i]:
            acc += 1
    acc = float(acc)/n
    return acc


batch_size = 16
dataset = textData(trainPath)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', return_dict=True)


# fine-tune the last four layers
train_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
for name, param in model.named_parameters():
    param.requires_grad = False
    for word in train_layers:
        if word in name:
            param.requires_grad = True
            break

model.to(device)
model.train()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

epoch_num = 10
total_step = len(dataset)/batch_size

# train
for epoch in range(epoch_num):
    for step, (lines, labels) in enumerate(dataloader):
        labels = labels.to(device)
        lines = tokenizer(lines, return_tensors='pt',
                          padding=True, truncation=True).to(device)
        input_ids = lines['input_ids']
        attention_mask = lines['attention_mask']

        outputs = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            acc = calc_acc(outputs, labels)
            print("epoch [%d/%d], step [%d/%d], loss %f, acc %.2f" %
                  (epoch+1, epoch_num, step, total_step, loss, acc))

    print("save model with epoch %d" % (epoch+1))
    torch.save(model, savePath)
