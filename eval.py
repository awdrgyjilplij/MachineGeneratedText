from pathlib import Path
from transformers import BertTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

dataPath = Path('data')
savePath = Path('model.pkl')
devPath = Path(dataPath, 'test1.txt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# calculate the accuracy, presicion, recall and F1 score 
def calc_score(output_labels, labels):
    n = len(output_labels)
    acc = 0
    pres = 0
    pres_den = 0
    recall = 0
    recall_den = 0
    for i in range(n):
        if output_labels[i] == labels[i]:
            acc += 1
        if output_labels[i] == 1:
            pres_den += 1
            if labels[i] == 1:
                pres += 1
        if labels[i] == 1:
            recall_den += 1
            if output_labels[i] == 1:
                recall += 1
    acc = float(acc)/n
    pres = float(pres)/pres_den
    recall = float(recall)/recall_den
    f = 2*pres*recall/(pres+recall)
    return acc, pres, recall, f

# load the model
model = torch.load(savePath)
model.to(device)
model.eval()
torch.no_grad()

lines = []
labels = []

with open(devPath, encoding='utf-8') as dev_f:
    for line in dev_f:
        line = line.strip()
        lines.append(line[:-1])
        labels.append(int(line[-1]))

Len=len(labels)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
output_labels=[]

# Due to the shortage of RAM, we process sixteen lines at a time
batch_size=16
for i in tqdm(range(0,Len,batch_size)): 
    temp_lines=lines[i:i+batch_size]
    temp_lines = tokenizer(temp_lines, return_tensors='pt',
                            padding=True, truncation=True).to(device)
    input_ids = temp_lines['input_ids']
    attention_mask = temp_lines['attention_mask']

    outputs = model(input_ids, attention_mask=attention_mask).logits

    for item in outputs:
        if item[0]>item[1]:
            output_labels.append(0)
        else:
            output_labels.append(1)

score=calc_score(output_labels, labels)
print("accuracy %.4f, presicion %.4f, recall %.4f and F1 score %.4f"%(score))
