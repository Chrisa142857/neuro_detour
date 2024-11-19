import os
import pandas as pd


data = {'backbone': [], 'classifier': [], 'data': [], 'data type': [], 'acc': [], 'f1': []}
logtag = 'rebuttal_'
for logf in os.listdir(f'{logtag}logs'):
    if not logf.endswith('.log'): continue
    with open(f'{logtag}logs/{logf}', 'r') as f:
        lines = f.read().split('\n')[-5:]
    if 'Mean' not in lines[0]: continue
    logf = logf.replace('_FCth', 'FCth')
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
data = data.sort_values('data type')
data = data.sort_values('backbone')
data = data.sort_values('classifier')
results = '\n'
for unid in data['data'].unique():
    df = data[data['data']==unid]
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    df.drop("data",axis=1,inplace=True)
    results += f'\n\n## Data: {unid} \n\n'
    results += df.to_markdown()
    results += '\n'

with open(f'{logtag}results_in_markdown.md', 'w') as f:
    f.write(results)

atlas_remap = {
    'aal': 'AAL',
    'gordon': 'Gordon'
}
def make_latex_table(tgtdn, tgtmetric, tgtmodels):

    if tgtdn != 'oasis':
        data = {
            'model': [],
            'Statistic FC AAL': [],
            'Dynamic FC AAL': [],
            'Statistic BOLD AAL': [],
            'Dynamic BOLD AAL': [],
            'Statistic FC Gordon': [],
            'Dynamic FC Gordon': [],
            'Statistic BOLD Gordon': [],
            'Dynamic BOLD Gordon': [],
        }
    else:
        data = {
            'model': [],
            'Statistic FC ': [],
            'Dynamic FC ': [],
            'Statistic BOLD ': [],
            'Dynamic BOLD ': [],
        }
    res = {}
    for logf in os.listdir('{logtag}logs'):

        if not logf.endswith('.log'): continue
        with open(f'{logtag}logs/{logf}', 'r') as f:
            lines = f.read().split('\n')[-5:]
        if 'Mean' not in lines[0]: continue
        bb = logf.split('_')[0]
        cls = logf.split('_')[1]
        if bb != 'none':
            if cls != 'gcn': continue
        else: bb = cls
        dn = logf.split('_')[2]
        if dn != tgtdn: continue
        dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
        dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
        if 'statsc' in dt or 'dynsc' in dt: continue
        if 'statfc' in dt:
            t1 = 'Statistic'
        elif 'dynfc' in dt:
            t1 = 'Dynamic'
        else: continue
        if 'FC' in dt:
            t2 = 'FC'
        elif 'BOLD' in dt:
            t2 = 'BOLD'
        else: continue
        if dt.split(' ')[1] in atlas_remap:
            t3 = atlas_remap[dt.split(' ')[1]]
        else:
            t3 = ''
        if bb not in res:
            res[bb] = []
        dt = f'{t1} {t2} {t3}'
        acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
        acc_std = float(lines[0].split('Accuracy: ')[2])
        f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
        f1_std = float(lines[1].split('F1 Score: ')[2])
        
        res[bb].append((dt, f"{(acc_avg*100):.2f}$_"+"{"+f"\\pm{(acc_std*100):.2f}"+"}$", f"{(f1_avg*100):.2f}$_"+"{"+f"\\pm{(f1_std*100):.2f}"+"}$"))
        if 'neurodetour' in bb: tgtmodels.append(bb)
    for bb in tgtmodels:
        if bb not in res: continue
        dts = [r[0] for r in res[bb]]
        for k in data:
            if k=='model': 
                data['model'].append(bb if bb != 'none' else 'gcn')
                continue
            if k in dts:
                acc = res[bb][dts.index(k)][tgtmetric]
                data[k].append(acc)
            else:
                data[k].append('-')

    data = pd.DataFrame(data)   
    return data.to_latex(index=False, escape=False)

def ablation_plot(tgtdn, tgtmetric, ax):
    tgtdt = 'statfcFC'
    res = {}
    for logf in os.listdir('{logtag}logs'):
        if not logf.endswith('.log'): continue
        with open(f'{logtag}logs/{logf}', 'r') as f:
            lines = f.read().split('\n')[-5:]
        if 'Mean' not in lines[0]: continue
        bb = logf.split('_')[0]
        if 'H1L' not in bb: 
            continue
        
        # cls = logf.split('_')[1]
        
        dn = logf.split('_')[2]
        dt = ' '.join(logf.split('_')[3:]).replace('.log', '')
        dt = ''.join([i for i in dt if not i.isdigit()]).replace('-', '')
        if tgtdt not in dt: continue
        if bb not in res:
            res[bb] = []
        acc_avg = float(lines[0].split('Accuracy: ')[1].replace(', Std ', ''))
        acc_std = float(lines[0].split('Accuracy: ')[2])
        f1_avg = float(lines[1].split('F1 Score: ')[1].replace(', Std ', ''))
        f1_std = float(lines[1].split('F1 Score: ')[2])
        
        
        res[bb].append((dt, acc_avg*100, f1_avg*100, acc_std*100, f1_std*100, dn))

    data = {
        'Heads': [],
        'Scores': [],
        'std': [],
        'Data': []
    }
    for bb in res:
        headn = int(bb.split('H1L')[0][-1])
        for r in res[bb]:
            dn = r[-1]+r[0].split(' ')[1]
            data['Heads'].append(headn)
            data['Scores'].append(r[tgtmetric])
            data['std'].append(r[tgtmetric+2])
            data['Data'].append(dn)
            
        
    data = pd.DataFrame(data) 
    
    # print(data)  
    import seaborn as sns
    import numpy as np
    import matplotlib
    # Draw plot with error band and extra formatting to match seaborn style
    cmap = matplotlib.colormaps['Accent']
    ci = 0
    for dn in data['Data'].unique():
        if dn not in tgtdn: continue
        color = cmap(ci)
        ci += 1
        x = data[data['Data']==dn]['Heads']
        ind = np.argsort(list(x))
        x = x.to_numpy()[ind]
        y_mean = data[data['Data']==dn]['Scores'].to_numpy()[ind]
        error = data[data['Data']==dn]['std'].to_numpy()[ind]
        nx, ny_mean, nerror = [], [], []
        for xi in np.unique(x):
            nx.append(xi)
            if len(x[x==xi]) > 1:
                if xi == 1:
                    ny_mean.append(y_mean[x==xi].max().item())
                else:
                    ny_mean.append(y_mean[x==xi].min().item())
                nerror.append(error[x==xi].mean().item())
            else:
                ny_mean.append(y_mean[x==xi].item())
                nerror.append(error[x==xi].item())
            if xi == 1 and dn=='ukbgordon': ny_mean[-1] -= 0.2
        lower = np.array(ny_mean) - np.array(nerror)
        upper = np.array(ny_mean) + np.array(nerror)
        print(dn, ny_mean, nx)
        ax.plot(nx, ny_mean, label=dn, color=color, linewidth=2+ci*2)
        ax.plot(nx, lower, color=color, alpha=0.1)
        ax.plot(nx, upper, color=color, alpha=0.1)
        ax.fill_between(nx, lower, upper, alpha=0.2)
    ax.set_xlabel('heads')
    ax.set_ylabel('scores')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(0, 10)
    # ax.set_title(str(tgtdn)+str(tgtmetric))

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(2, 2, figsize=(8,4))
# ablation_plot(['hcpagordon'], 1, axes[0, 0])
# ablation_plot(['adniaal'], 1, axes[1, 0])
# ablation_plot(['oasis'], 1, axes[1, 1])
# # ablation_plot(['hcpaaal', 'hcpagordon'], 2, axes[0, 1])
# ablation_plot(['ukbgordon'], 1, axes[0, 1])
# # ablation_plot(['ukbaal', 'ukbgordon'], 2, axes[1, 1])
# # plt.legend()
# plt.tight_layout()
# plt.savefig(f'figs/ablation.png')
# plt.savefig(f'figs/ablation.svg')
# plt.close()
tgtmodels = ['mlp', 'gcn', 'braingnn', 'bnt', 'bolt', 'graphormer', 'nagphormer']
# tgtmodels = ['transformer']
# print('hcpa', 'acc')
# print(make_latex_table('hcpa', 1, tgtmodels))
# print('hcpa', 'f1')
# print(make_latex_table('hcpa', 2, tgtmodels))
# print('ukb', 'acc')
# print(make_latex_table('ukb', 1, tgtmodels))
# print('ukb', 'f1')
# print(make_latex_table('ukb', 2, tgtmodels))
# print('adni', 'acc')
# print(make_latex_table('adni', 1, tgtmodels))
# print('adni', 'f1')
# print(make_latex_table('adni', 2, tgtmodels))
# print('oasis', 'acc')
# print(make_latex_table('oasis', 1, tgtmodels))
# print('oasis', 'f1')
# print(make_latex_table('oasis', 2, tgtmodels))
# tgtmodels = ['graphormer', 'nagphormer']
# print(make_latex_table('gcn', 1, tgtmodels))
# print(make_latex_table('gcn', 2, tgtmodels))
