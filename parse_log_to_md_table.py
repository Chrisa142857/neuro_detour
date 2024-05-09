import os
import pandas as pd


data = {'backbone': [], 'classifier': [], 'data': [], 'data type': [], 'acc': [], 'f1': []}

for logf in os.listdir('logs'):
    if not logf.endswith('.log'): continue
    with open(f'logs/{logf}', 'r') as f:
        lines = f.read().split('\n')[-5:]
    if 'Mean' not in lines[0]: continue
    bb = logf.split('_')[0]
    cls = logf.split('_')[1]
    dn = logf.split('_')[2]
    dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
    dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
    if 'statsc' in dt or 'dynsc' in dt: continue
    acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
    acc_std = float(lines[0].split('Accuracy: ')[2])
    f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
    f1_std = float(lines[1].split('F1 Score: ')[2])
    data['backbone'].append(bb)
    data['classifier'].append(cls)
    data['data'].append(dn)
    data['data type'].append(dt)
    data['acc'].append(f"{acc_avg:.5f}+-{acc_std:.5f}")
    data['f1'].append(f"{f1_avg:.5f}+-{f1_std:.5f}")

data = pd.DataFrame(data)
data = data.sort_values('classifier')
data = data.sort_values('data type')
data = data.sort_values('backbone')
results = '\n'
for unid in data['data'].unique():
    df = data[data['data']==unid]
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.drop("data",axis=1,inplace=True)
    results += f'\n\n## Data: {unid} \n\n'
    results += df.to_markdown()
    results += '\n'

with open('results_in_markdown.md', 'w') as f:
    f.write(results)
