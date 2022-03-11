import json
import pandas as pd

with open('train.json', 'r', encoding='utf-8')as f:
    train_data = json.load(f)

train_df = []

for i in range(len(train_data)):
    data = train_data[i]
    content = data['Content']
    questions = data['Questions']
    for question in questions:
        question['Content'] = content
        train_df.append(question)

train_df =  pd.DataFrame(train_df)

with open('validation.json', 'r', encoding='utf-8')as f:
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