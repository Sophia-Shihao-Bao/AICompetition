# -*- coding:utf-8 -*-
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *
from argparse import ArgumentParser

CFG = { #训练的参数配置
    'fold_num': 10, #五折交叉验证
    'seed': 42,
    'model': 'hfl/chinese-bert-wwm-ext', #预训练模型
    'max_len': 230, #文本截断的最大长度
    'epochs': 10,
    'train_bs': 14, #batch_size，可根据自己的显存调整
    'valid_bs': 14,
    'lr': 2e-5, #学习率
    'num_workers': 8,
    'accum_iter': 2, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser()
    parser.add_argument("-input", type=str, required=True)
    parser.add_argument("-output", type=str, required=True)
    args = parser.parse_args()

with open('args.input', 'r', encoding='utf-8')as f:
    test_data = json.load(f)

test_df = []

for i in range(len(test_data)):
    data = test_data[i]
    content = data['Content']
    questions = data['Questions']
    cls = data['Type']
    diff = data['Diff']
    for question in questions:
        question['Content'] = content
        question['Type'] = cls
        question['Diff'] = diff
        test_df.append(question)

test_df = pd.DataFrame(test_df)

#train_df.to_csv('train.csv',index=False)
test_df.to_csv('test.csv',index=False)

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df =  pd.read_csv('train.csv')
test_df =  pd.read_csv('test.csv')
train_df['label'] = train_df['Answer'].apply(lambda x:['A','B','C','D'].index(x)) #将标签从ABCD转成0123
test_df['label'] = 0

tokenizer = BertTokenizer.from_pretrained(CFG['model']) #加载bert的分词器


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):  # 将一条数据从(文章,问题,4个选项)转成(文章,问题,选项1)、(文章,问题,选项2)...
        label = self.df.label.values[idx]
        question = self.df.Question.values[idx]
        content = self.df.Content.values[idx]
        choice = self.df.Choices.values[idx][2:-2].split('\', \'')
        if len(choice) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choice)):
                choice.append('D．不知道')

        content = [content for i in range(len(choice))]
        pair = [question + ' ' + i[2:] for i in choice]

        return content, pair, label

def collate_fn(data): #将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True, max_length=CFG['max_len'], return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[-1] for x in data])
    return input_ids, attention_mask, token_type_ids, label


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), y.to(device).long()

        with autocast():  # 使用半精度训练
            output = model(input_ids, attention_mask, token_type_ids).logits

            loss = criterion(output, y) / CFG['accum_iter']
            scaler.scale(loss).backward()

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        losses.update(loss.item() * CFG['accum_iter'], y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


seed_everything(CFG['seed'])

folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']) \
    .split(np.arange(train_df.shape[0]), train_df.label.values)  # 五折交叉验证

cv = []  # 保存每折的最佳准确率


test_set = MyDataset(test_df)
test_loader = DataLoader(test_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False, num_workers=CFG['num_workers'])

model = BertForMultipleChoice.from_pretrained(CFG['model']).to(device)

predictions = []

#for fold in [0, 1, 2, 3, 4]:  # 把训练后的五个模型挨个进行预测
for fold in [0,1]:
    y_pred = []
    model.load_state_dict(torch.load('{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold)))

    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(device), attention_mask.to(
                device), token_type_ids.to(device), y.to(device).long()

            output = model(input_ids, attention_mask, token_type_ids).logits.cpu().numpy()

            y_pred.extend(output)

    predictions += [y_pred]
    print(predictions)

predictions = np.mean(predictions,0).argmax(1) #将结果按五折进行平均，然后argmax得到label
sub = pd.read_csv('sample.csv',dtype=object) #提交
sub['label'] = predictions
sub['label'] = sub['label'].apply(lambda x:['A','B','C','D'][x])

sub.to_csv('args.output',index=False)

np.mean(cv)