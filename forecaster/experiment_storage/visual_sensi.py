import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter, LinearLocator, FormatStrFormatter

current_directory = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300
})


def extract_value(data_str):
    if isinstance(data_str, str):
        return float(data_str[:5])
    return float(data_str)


def plot_metrics_by_experiment(experiment: str, save: bool = False):
    assert experiment in ("anchor_size", "seen_node_ratio")

    excel_path = os.path.join(current_directory, f"experiment_results_{experiment}.xlsx")
    sheet_all = "all"
    sheet_unseen = "unseen"
    xlabel = "Anchor Size" if experiment == "anchor_size" else "Training Node Ratio"

    df_all = pd.read_excel(excel_path, sheet_name=sheet_all, usecols="B:D", header=0)
    df_unseen = pd.read_excel(excel_path, sheet_name=sheet_unseen, usecols="B:D", header=0)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df.iloc[:, 1] = df.iloc[:, 1].apply(extract_value)
        df.iloc[:, 2] = df.iloc[:, 2].apply(extract_value)
        df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
        return df

    df_all = clean(df_all)
    df_unseen = clean(df_unseen)

    def split_blocks(df: pd.DataFrame):
        x = df.iloc[:, 0].to_numpy()
        bounds = [0]
        for i in range(1, len(x)):
            if x[i] < x[i - 1]:
                bounds.append(i)
        bounds.append(len(x))
        return [df.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True) for k in range(len(bounds) - 1)]

    blocks_all = split_blocks(df_all)
    blocks_unseen = split_blocks(df_unseen)

    titles = ["PEMS03", "PEMS04", "PEMS08"]
    mae_color = "#bc5c58"
    rmse_color = "#7293c2"

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    legend_handles = []
    legend_labels = []

    for k in range(3):
        ax_mae = axes[k]
        ax_rmse = ax_mae.twinx()
        a = blocks_all[k]
        u = blocks_unseen[k]

        l1, = ax_mae.plot(a.iloc[:, 0], a.iloc[:, 1], color=mae_color, linestyle='-', marker='o', markersize=3,
                          label='MAE (All)' if k == 0 else '_nolegend_')
        l2, = ax_rmse.plot(a.iloc[:, 0], a.iloc[:, 2], color=rmse_color, linestyle='-', marker='s', markersize=3,
                           label='RMSE (All)' if k == 0 else '_nolegend_')
        if experiment == "seen_node_ratio":
            l3, = ax_mae.plot(u.iloc[:-1, 0], u.iloc[:-1, 1], color=mae_color, linestyle='--', marker='o', markersize=3,
                              label='MAE (Inaccessible)' if k == 0 else '_nolegend_')
            l4, = ax_rmse.plot(u.iloc[:-1, 0], u.iloc[:-1, 2], color=rmse_color, linestyle='--', marker='s', markersize=3,
                               label='RMSE (Inaccessible)' if k == 0 else '_nolegend_')
        else:
            l3, = ax_mae.plot(u.iloc[:, 0], u.iloc[:, 1], color=mae_color, linestyle='--', marker='o', markersize=3,
                              label='MAE (Inaccessible)' if k == 0 else '_nolegend_')
            l4, = ax_rmse.plot(u.iloc[:, 0], u.iloc[:, 2], color=rmse_color, linestyle='--', marker='s', markersize=3,
                               label='RMSE (Inaccessible)' if k == 0 else '_nolegend_')

        if k == 0:
            legend_handles = [l1, l2, l3, l4]
            legend_labels = [h.get_label() for h in legend_handles]

        u_mae_series = u.iloc[:-1, 1] if experiment == "seen_node_ratio" else u.iloc[:, 1]
        mae_min = float(min(a.iloc[:, 1].min(), u_mae_series.min()))
        mae_max = float(max(a.iloc[:, 1].max(), u_mae_series.max()))
        mae_pad = 0.4 * (mae_max - mae_min)
        ax_mae.set_ylim(mae_min - mae_pad * 0.2, mae_max + mae_pad)

        u_rmse_series = u.iloc[:-1, 2] if experiment == "seen_node_ratio" else u.iloc[:, 2]
        rmse_min = float(min(a.iloc[:, 2].min(), u_rmse_series.min()))
        rmse_max = float(max(a.iloc[:, 2].max(), u_rmse_series.max()))
        rmse_pad = 0.4 * (rmse_max - rmse_min)
        ax_rmse.set_ylim(rmse_min - rmse_pad, rmse_max + rmse_pad * 0.2)

        ax_mae.yaxis.set_major_locator(LinearLocator(6))
        ax_rmse.yaxis.set_major_locator(LinearLocator(6))
        ax_mae.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_rmse.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax_mae.tick_params(axis='y', colors=mae_color)
        ax_rmse.tick_params(axis='y', colors=rmse_color)

        if experiment == "seen_node_ratio":
            ax_mae.xaxis.set_major_formatter(PercentFormatter(1.0))

        ax_mae.grid(True, linestyle='--', alpha=0.3)
        ax_mae.set_title(titles[k], fontsize=10)
        if k == 1:
            ax_mae.set_xlabel(xlabel, fontweight='bold')

    fig.text(0.005, 0.5, 'MAE', va='center', rotation='vertical', color=mae_color, fontweight='bold')
    fig.text(0.995, 0.5, 'RMSE', va='center', rotation='vertical', color=rmse_color, fontweight='bold')

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.06))
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.27)

    if save:
        outname = f"metrics_triptych_{experiment}"
        outpath = os.path.join(current_directory, outname)
        plt.savefig(f"{outpath}.pdf", format='pdf', bbox_inches='tight')
        plt.savefig(f"{outpath}.png", dpi=300, bbox_inches='tight')

    return fig, axes


def plot_dual_experiments():
    anchor_file = os.path.join(current_directory, "experiment_results_anchor_size.xlsx")
    seen_file = os.path.join(current_directory, "experiment_results_seen_node_ratio.xlsx")

    anchor_all = pd.read_excel(anchor_file, sheet_name='all', usecols='B:D', skiprows=1, nrows=8, header=None)
    anchor_unseen = pd.read_excel(anchor_file, sheet_name='unseen', usecols='B:D', skiprows=1, nrows=8, header=None)

    seen_all = pd.read_excel(seen_file, sheet_name='all', usecols='B:D', skiprows=13, nrows=6, header=None)
    seen_unseen = pd.read_excel(seen_file, sheet_name='unseen', usecols='B:D', skiprows=13, nrows=6, header=None)

    anchor_all.iloc[:, 2] = anchor_all.iloc[:, 2].apply(extract_value)
    anchor_unseen.iloc[:, 2] = anchor_unseen.iloc[:, 2].apply(extract_value)
    seen_all.iloc[:, 2] = seen_all.iloc[:, 2].apply(extract_value)
    seen_unseen.iloc[:, 2] = seen_unseen.iloc[:, 2].apply(extract_value)
    print(anchor_all)
    print(seen_all)

    fig, ax1 = plt.subplots(figsize=(4, 3))

    anchor_color = '#bc5c58'
    seen_color = '#7293c2'

    line1, = ax1.plot(anchor_all.iloc[:, 0], anchor_all.iloc[:, 2],
                      color=anchor_color, linestyle='-', marker='o', markersize=4,
                      label='Anchor Size (All)')
    line2, = ax1.plot(anchor_unseen.iloc[:, 0], anchor_unseen.iloc[:, 2],
                      color=anchor_color, linestyle='--', marker='o', markersize=4,
                      label='Anchor Size (Inaccessible)')

    ax2 = ax1.twiny()

    line3, = ax2.plot(seen_all.iloc[:, 0], seen_all.iloc[:, 2],
                      color=seen_color, linestyle='-', marker='s', markersize=4,
                      label='Training Node Ratio (All)')
    line4, = ax2.plot(seen_unseen.iloc[:-1, 0], seen_unseen.iloc[:-1, 2],
                      color=seen_color, linestyle='--', marker='s', markersize=4,
                      label='Training Node Ratio (Inaccessible)')

    ax1.set_xlabel('Anchor Size', color=anchor_color, fontweight='bold')
    ax2.set_xlabel('Training Node Ratio', color=seen_color, fontweight='bold')
    ax1.set_ylabel('RMSE', fontweight='bold')

    ax1.tick_params(axis='x', labelcolor=anchor_color)
    ax2.tick_params(axis='x', labelcolor=seen_color)
    ax1.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
    ax2.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax2.xaxis.set_major_formatter(PercentFormatter(1.0))

    ax1.grid(True, linestyle='--', alpha=0.3)

    lines = [line1, line2, line3, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=False)

    plt.tight_layout()

    return fig, ax1, ax2


# fig, ax1, ax2 = plot_dual_experiments()

# output_path = f"{current_directory}/dual_experiment_comparison"
# plt.savefig(f"{output_path}.pdf", format='pdf', bbox_inches='tight')
# plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')

# plt.show()

plot_metrics_by_experiment(experiment="anchor_size", save=True)
plot_metrics_by_experiment(experiment="seen_node_ratio", save=True)
