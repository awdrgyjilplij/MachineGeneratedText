from pathlib import Path
from transformers import BertTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

dataPath = Path('data')
savePath = Path('model.pkl')
outPath = Path('result.txt')
testPath=Path(dataPath,'test2_no_label.txt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load the model
model = torch.load(savePath)
model.to(device)
model.eval()
torch.no_grad()

lines = []
with open(testPath, encoding='utf-8') as dev_f:
    for line in dev_f:
        line = line.strip()
        lines.append(line[:-1])

Len=len(lines)

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

with open(outPath,'w',encoding='utf-8') as out_f:
    for i in output_labels:
        out_f.write("%d\n"%(i))