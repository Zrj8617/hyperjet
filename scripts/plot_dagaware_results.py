import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = '/data2/zrj2025/各类文档/result_graph/2026_5_10_dagaware_teacher'
OUT_DIR = os.path.join(RESULT_DIR, 'generated_plots')
os.makedirs(OUT_DIR, exist_ok=True)

# files to load
files = [
    'log_data_dagaware_uav8_arr0050_full_seed42_2026-05-10.json',
    'log_data_dagaware_uav8_arr0050_fallback_seed42_2026-05-10.json',
    'log_data_dagaware_uav8_arr0050_full_seed43_2026-05-10.json',
    'log_data_dagaware_uav8_arr0050_fallback_seed43_2026-05-10.json',
    'log_data_dagaware_uav8_arr0050_full_seed44_2026-05-10.json',
    'log_data_dagaware_uav8_arr0050_fallback_seed44_2026-05-10.json',
]
labels = [
    'full_s42','fallback_s42','full_s43','fallback_s43','full_s44','fallback_s44'
]

data = {}
for f,label in zip(files, labels):
    path = os.path.join(RESULT_DIR, f)
    if not os.path.exists(path):
        print('Missing', path)
        continue
    with open(path,'r') as fh:
        arr = json.load(fh)
    data[label] = arr

# common metrics
metrics = ['reward','latency','energy','fairness','offline_rate']

# helper to extract series
def series(arr, key):
    return [entry.get(key, np.nan) for entry in arr]

# plot each metric
for metric in metrics:
    plt.figure(figsize=(10,6))
    for label, arr in data.items():
        x = [e.get('episode', i+1) for i,e in enumerate(arr)]
        y = series(arr, metric)
        plt.plot(x, y, label=label, linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.title(metric + ' comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'compare_{metric}.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print('Saved', out)

# plot DAG success rate and task drop/finish rates
dag_metrics = ['dag_success_rate','dag_task_finish_rate','dag_task_drop_rate']
for metric in dag_metrics:
    plt.figure(figsize=(10,6))
    for label, arr in data.items():
        x = [e.get('episode', i+1) for i,e in enumerate(arr)]
        y = series(arr, metric)
        plt.plot(x, y, label=label, linewidth=1.5)
    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.title(metric + ' comparison')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'compare_{metric}.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print('Saved', out)

# summary table: compute mean of last 5 episodes for each label
summary = {}
for label, arr in data.items():
    if not arr:
        continue
    lastk = arr[-5:]
    summary[label] = {m: float(np.nanmean([e.get(m, np.nan) for e in lastk])) for m in metrics + dag_metrics}

# save summary
with open(os.path.join(OUT_DIR,'summary_last5.json'),'w') as fh:
    json.dump(summary, fh, indent=4)
print('Saved summary')

# per-seed full vs fallback plots
seeds = ['42', '43', '44']
for s in seeds:
    for metric in metrics:
        plt.figure(figsize=(8,5))
        for mode in ['full', 'fallback']:
            label = f'{mode}_s{s}'
            arr = data.get(label)
            if not arr:
                continue
            x = [e.get('episode', i+1) for i,e in enumerate(arr)]
            y = series(arr, metric)
            plt.plot(x, y, label=mode, linewidth=1.8)
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.title(f'seed{s} {metric} full vs fallback')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f'seed{s}_compare_{metric}.png')
        plt.savefig(out, dpi=200)
        plt.close()
        print('Saved', out)
