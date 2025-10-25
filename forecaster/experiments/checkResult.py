import glob
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def format_mean_std(values):
    if len(values) == 0:
        return 'N/A'
    mean_val = np.mean(values)
    if len(values) > 1:
        std_val = np.std(values, ddof=1)
        return f'{mean_val:.2f}(±{std_val:.2f})'
    else:
        return f'{mean_val:.2f}'


if __name__ == '__main__':
    metric_files = glob.glob(os.path.join('experiment_storage', 'anchor_size', '**', 'metrics.npy'), recursive=True)
    print(f"Find {len(metric_files)} metrics.npy files")

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    models = set()

    for f in metric_files:
        path_parts = list(Path(f).parts)
        exp_name = path_parts[1]

        # ['experiment_storage', 'comp', 'PEMS04', 'metast', 'default', 'seed=42', 'metrics.npy']
        if exp_name == 'comp':
            dataset = path_parts[2]  # PEMS03, PEMS04, PEMS08
            model = path_parts[3]    # metast, nbeats
            seed = path_parts[5]     # seed=42, seed=123, seed=456

        # ['experiment_storage', 'ablation', 'PEMS04', 'metast', 'condition_type=attention', 'seed=42', 'metrics.npy']
        elif exp_name == 'ablation':
            dataset = path_parts[2]  # PEMS03, PEMS04, PEMS08
            model = path_parts[4]    # condition_type=
            seed = path_parts[5]     # seed=42, seed=123, seed=456

        # ['experiment_storage', 'case_study', 'water_mn', 'metast', 'default', 'seed=42', 'metrics.npy']
        elif exp_name == 'case_study':
            dataset = path_parts[2]  # water_mn, water_o2
            model = path_parts[3]    # metast
            seed = path_parts[5]     # seed=42, seed=123, seed=456

        # ['experiment_storage', 'anchor_size', 'PEMS08', 'metast', 'k_neighbors', 'seed=42', 'metrics.npy']
        elif exp_name == 'anchor_size':
            dataset = path_parts[2]  # PEMS08
            model = path_parts[4]    # k_neighbors
            seed = path_parts[5]     # seed=42, seed=123, seed=456

        # ['experiment_storage', 'seen_node_ratio', 'seen_node_ratio005', 'PEMS03', 'metast', 'default', 'seed=42', 'metrics.npy']
        elif exp_name == 'seen_node_ratio':
            dataset = path_parts[3]  # PEMS03
            model = path_parts[2]    # seen_node_ratio005
            seed = path_parts[6]     # seed=42, seed=123, seed=456

        else:
            raise ValueError("exp_name error")

        models.add(model)

        metrics = np.load(f, allow_pickle=True).item()
        test_metrics = metrics['test']

        results[dataset][model]['mae'].append(test_metrics['mae'])
        results[dataset][model]['rmse'].append(test_metrics['rmse'])
        results[dataset][model]['mape'].append(test_metrics['mape'] * 100)

    datasets = sorted(results.keys())
    models = sorted(list(models))
    print('datasets: ', datasets)
    print('models: ', models)

    index_data = []
    table_data = []

    for dataset in datasets:
        for model in models:
            index_data.append((dataset, model))
            if model in results[dataset]:
                mae_str = format_mean_std(results[dataset][model]['mae'])
                rmse_str = format_mean_std(results[dataset][model]['rmse'])
                mape_str = format_mean_std(results[dataset][model]['mape'])
            else:
                mae_str = rmse_str = mape_str = 'N/A'

            table_data.append([mae_str, rmse_str, mape_str])

    multi_index = pd.MultiIndex.from_tuples(index_data, names=['Dataset', 'Model'])
    df_advanced = pd.DataFrame(table_data,
                               index=multi_index,
                               columns=['MAE', 'RMSE', 'MAPE'])

    print(df_advanced.to_string())

    output_file = os.path.join('experiment_storage', f'experiment_results_{exp_name}.xlsx')
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_advanced.to_excel(writer, sheet_name='Advanced_Format')
    # output_file = os.path.join('experiment_storage', f'experiment_results_{exp_name}.csv')
    # df_advanced.to_csv(output_file, index=False)

    print(f'\nsaved as: {output_file}')
