import os
import csv
import math
from typing import Dict, List, Tuple, Optional

import mlflow
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mlflow import MlflowClient


# MLflow configuration

MLFLOW_DB_PATH = "../mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
EXPERIMENT_NAME = "Deep Learning Project"

OUTPUT_DIR = "optimizer_figures_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
assert experiment is not None, f"Experiment not found: {EXPERIMENT_NAME}"


# metrics & params

TRAIN_LOSS_KEY = "train/loss"
VAL_LOSS_KEY = "val/loss"

TRAIN_F1_KEY = "train/f1_macro"
VAL_F1_KEY = "val/f1_macro"

OPTIM_PARAM_KEY = "optim"
LR_PARAM_KEY = "lr"
WEIGHT_DECAY_PARAM_KEY = "weight_decay"


# GPU memory metric
GPU_MEMORY_MB_CANDIDATES = [
    "system/gpu_0_memory_usage_megabytes",
]


# MLflow filter strings

FINAL_COMPARISON_FILTER = """
    tags.study = 'Final'
"""


# retrieve AdamW learning-rate sweep runs only
# multiple AdamW runs with different params.lr values
ADAMW_LR_SWEEP_FILTER = """
    tags.study = 'Ablation'
    AND tags.ablation_factor = 'LR'
    AND params.optim = 'AdamW'
"""


# retrieve Muon learning-rate sweep runs only
# multiple Muon runs with different params.lr values
MUON_LR_SWEEP_FILTER = """
    tags.study = 'Ablation'
    AND tags.ablation_factor = 'LR'
    AND params.optim = 'Muon'
"""


# retrieve NorMuon learning-rate sweep runs only
# multiple NorMuon runs with different params.lr values
NORMUON_LR_SWEEP_FILTER = """
    tags.study = 'Ablation'
    AND tags.ablation_factor = 'LR'
    AND params.optim = 'NorMuon'
"""


LR_SWEEP_FILTERS = {
    "AdamW": ADAMW_LR_SWEEP_FILTER,
    "Muon": MUON_LR_SWEEP_FILTER,
    "NorMuon": NORMUON_LR_SWEEP_FILTER,
}


# learning rates used in the final comparison
# used only to make the chosen LR curve thicker in LR sweep plots
SELECTED_LR_BY_OPTIM = {
    "AdamW": 6.15e-5,
    "Muon": 1.82e-3,
    "NorMuon": 2.67e-4,
}


# visual configuration

OPTIMIZER_ORDER = [
    "AdamW",
    "Muon",
    "NorMuon",
    "MuonCom",
    "NorMuonCom",
]

OPTIMIZER_DISPLAY_NAMES = {
    "AdamW": "AdamW",
    "Muon": "Muon",
    "NorMuon": "NorMuon",
    "MuonCom": "Muon Combined",
    "NorMuonCom": "NorMuon Combined",
}

OPTIMIZER_COLORS = {
    "AdamW": "#1f77b4",
    "Muon": "#ff7f0e",
    "NorMuon": "#2ca02c",
    "MuonCom": "#d62728",
    "NorMuonCom": "#9467bd",
}

OPTIMIZER_LINESTYLES = {
    "AdamW": "-",
    "Muon": "--",
    "NorMuon": "-.",
    "MuonCom": ":",
    "NorMuonCom": (0, (5, 2)),
}

MAX_FINAL_STEP = 200
MAX_LR_SWEEP_STEP = 200

# Set to 1 to show exact raw curves
# Set to 3 or 5 if validation curves are too noisy
SMOOTHING_WINDOW_BY_METRIC = {
    TRAIN_LOSS_KEY: 1,
    VAL_LOSS_KEY: 3,
    TRAIN_F1_KEY: 1,
    VAL_F1_KEY: 3,
}

DPI = 300

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# general helper functions

def save_figure(filename: str) -> None:
    path_png = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()
    plt.savefig(path_png, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path_png}")


def get_runs(filter_string: str, max_results: int = 1000):
    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=["attributes.start_time ASC"],
    )


def get_optimizer_key(run) -> str:
    return run.data.params.get(OPTIM_PARAM_KEY, "Unknown")


def get_optimizer_display_name(run) -> str:
    optim_key = get_optimizer_key(run)
    return OPTIMIZER_DISPLAY_NAMES.get(optim_key, optim_key)


def get_float_param(run, param_key: str, default: Optional[float] = None) -> Optional[float]:
    raw_value = run.data.params.get(param_key)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def get_metric_xy(
    run,
    metric_key: str,
    max_step: Optional[int] = None,
) -> Tuple[List[int], List[float]]:

    history = client.get_metric_history(run.info.run_id, metric_key)

    if len(history) == 0:
        return [], []

    values_by_step = {}

    for point in history:
        if point.value is None:
            continue

        value = float(point.value)

        if not math.isfinite(value):
            continue

        step = int(point.step)

        if max_step is not None and step > max_step:
            continue

        # Keep latest logged value for duplicate steps.
        values_by_step[step] = value

    steps = sorted(values_by_step.keys())
    values = [values_by_step[step] for step in steps]

    return steps, values


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 2:
        return values

    smoothed = []

    for index in range(len(values)):
        start_index = max(0, index - window + 1)
        current_window = values[start_index:index + 1]
        smoothed.append(sum(current_window) / len(current_window))

    return smoothed


def get_best_metric_point(
    run,
    metric_key: str,
    higher_is_better: bool = True,
    max_step: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float]]:

    steps, values = get_metric_xy(run, metric_key, max_step=max_step)

    if len(values) == 0:
        return None, None

    if higher_is_better:
        best_index = max(range(len(values)), key=lambda i: values[i])
    else:
        best_index = min(range(len(values)), key=lambda i: values[i])

    return steps[best_index], values[best_index]


def get_last_metric_value(
    run,
    metric_key: str,
    max_step: Optional[int] = None,
) -> Optional[float]:

    steps, values = get_metric_xy(run, metric_key, max_step=max_step)

    if len(values) == 0:
        return None

    return values[-1]


# memory metric detection

def get_available_metric_keys(run) -> List[str]:
    full_run = client.get_run(run.info.run_id)
    return sorted(full_run.data.metrics.keys())


def resolve_gpu_memory_metric_key(run) -> Optional[str]:
    available_keys = get_available_metric_keys(run)
    available_key_set = set(available_keys)

    for candidate_key in GPU_MEMORY_MB_CANDIDATES:
        if candidate_key in available_key_set:
            return candidate_key

    for key in available_keys:
        lowered = key.lower()

        if "gpu" in lowered and "memory" in lowered:
            return key

    return None


def convert_memory_to_gb(metric_key: str, values: List[float]) -> List[float]:
    lowered = metric_key.lower()

    if "byte" in lowered and "mega" not in lowered and "mb" not in lowered:
        return [value / (1024.0 ** 3) for value in values]

    if "gb" in lowered or "gigabyte" in lowered:
        return values

    return [value / 1024.0 for value in values]


def get_gpu_memory_gb_values(run) -> Tuple[Optional[str], List[float]]:
    memory_key = resolve_gpu_memory_metric_key(run)

    if memory_key is None:
        return None, []

    _, memory_values = get_metric_xy(run, memory_key, max_step=None)

    if len(memory_values) == 0:
        return memory_key, []

    return memory_key, convert_memory_to_gb(memory_key, memory_values)


# run selection and summary

def select_final_runs(runs) -> List:
    """
    Keep one final run per optimizer.

    If the final filter return exactly one run per optimizer, simply
    return those runs in a fixed order.

    If it returns multiple runs per optimizer, keep the run with the
    best validation Macro-F1.
    """

    best_runs_by_optimizer = {}

    for run in runs:
        optim_key = get_optimizer_key(run)

        if optim_key not in OPTIMIZER_DISPLAY_NAMES:
            print(f"Skipping run with unknown optimizer: {optim_key}")
            continue

        _, best_val_f1 = get_best_metric_point(
            run,
            VAL_F1_KEY,
            higher_is_better=True,
            max_step=None,
        )

        if best_val_f1 is None:
            print(f"Skipping {optim_key}: no {VAL_F1_KEY}")
            continue

        current = best_runs_by_optimizer.get(optim_key)

        if current is None or best_val_f1 > current["best_val_f1"]:
            best_runs_by_optimizer[optim_key] = {
                "run": run,
                "best_val_f1": best_val_f1,
            }

    selected_runs = []

    for optim_key in OPTIMIZER_ORDER:
        if optim_key in best_runs_by_optimizer:
            selected_runs.append(best_runs_by_optimizer[optim_key]["run"])

    return selected_runs


def build_summary_rows(runs) -> List[Dict]:
    rows = []

    for run in runs:
        optim_key = get_optimizer_key(run)
        display_name = get_optimizer_display_name(run)

        best_val_f1_step, best_val_f1 = get_best_metric_point(
            run,
            VAL_F1_KEY,
            higher_is_better=True,
            max_step=None,
        )

        final_train_f1 = get_last_metric_value(
            run,
            TRAIN_F1_KEY,
            max_step=MAX_FINAL_STEP,
        )

        final_val_loss = get_last_metric_value(
            run,
            VAL_LOSS_KEY,
            max_step=MAX_FINAL_STEP,
        )

        memory_key, memory_gb_values = get_gpu_memory_gb_values(run)

        if len(memory_gb_values) > 0:
            peak_memory_gb = max(memory_gb_values)
            steady_memory_gb = sum(memory_gb_values[-5:]) / len(memory_gb_values[-5:])
        else:
            peak_memory_gb = None
            steady_memory_gb = None

        rows.append({
            "optimizer_key": optim_key,
            "optimizer": display_name,
            "lr": get_float_param(run, LR_PARAM_KEY),
            "weight_decay": get_float_param(run, WEIGHT_DECAY_PARAM_KEY),
            "best_val_f1": best_val_f1,
            "step_at_best_val_f1": best_val_f1_step,
            "final_train_f1": final_train_f1,
            "final_val_loss": final_val_loss,
            "gpu_memory_metric_key": memory_key,
            "peak_gpu_memory_gb": peak_memory_gb,
            "steady_gpu_memory_gb": steady_memory_gb,
            "run_id": run.info.run_id,
        })

    return rows


def save_summary_csv(rows: List[Dict], filename: str = "optimizer_summary.csv") -> None:
    path = os.path.join(OUTPUT_DIR, filename)

    fieldnames = [
        "optimizer",
        "lr",
        "weight_decay",
        "best_val_f1",
        "step_at_best_val_f1",
        "final_train_f1",
        "final_val_loss",
        "gpu_memory_metric_key",
        "peak_gpu_memory_gb",
        "steady_gpu_memory_gb",
        "run_id",
    ]

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Saved: {path}")


# compact direct-label line plots

def add_direct_labels(
    ax,
    label_items: List[Dict],
    x_position: float,
    min_gap_fraction: float = 0.075,
    connector_alpha: float = 0.45,
) -> None:
    """
    Add direct labels on the right side of a plot.

    Each label is vertically separated, and a light connector line is drawn
    from the actual curve endpoint to the label.
    """

    if len(label_items) == 0:
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    if y_range <= 0:
        return

    min_gap = y_range * min_gap_fraction
    top_limit = y_max - 0.03 * y_range
    bottom_limit = y_min + 0.03 * y_range

    # Sort top-to-bottom so high-ending curves are labeled first.
    sorted_items = sorted(
        label_items,
        key=lambda item: item["y"],
        reverse=True,
    )

    adjusted_items = []

    for item in sorted_items:
        proposed_y = item["y"]

        if len(adjusted_items) == 0:
            adjusted_y = min(proposed_y, top_limit)
        else:
            previous_y = adjusted_items[-1]["adjusted_y"]
            adjusted_y = min(proposed_y, previous_y - min_gap)

        adjusted_item = dict(item)
        adjusted_item["adjusted_y"] = adjusted_y
        adjusted_items.append(adjusted_item)

    # If the stack drops below the axis bottom, shift all labels up.
    lowest_label_y = adjusted_items[-1]["adjusted_y"]

    if lowest_label_y < bottom_limit:
        upward_shift = bottom_limit - lowest_label_y

        for item in adjusted_items:
            item["adjusted_y"] += upward_shift

    # If shifting upward pushed the top label above the axis, expand ylim.
    highest_label_y = adjusted_items[0]["adjusted_y"]

    if highest_label_y > top_limit:
        new_y_max = highest_label_y + 0.05 * y_range
        ax.set_ylim(y_min, new_y_max)

    for item in adjusted_items:
        curve_end_x = item["x"]
        curve_end_y = item["y"]
        label_y = item["adjusted_y"]
        color = item["color"]

        # Connector from actual curve end to shifted label.
        ax.plot(
            [curve_end_x, x_position * 0.985],
            [curve_end_y, label_y],
            color=color,
            linewidth=0.8,
            alpha=connector_alpha,
            clip_on=False,
        )

        ax.text(
            x_position,
            label_y,
            item["label"],
            color=color,
            fontsize=10,
            va="center",
            ha="left",
            clip_on=False,
        )


def plot_compact_metric_curves(
    runs,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
    max_step: int = MAX_FINAL_STEP,
) -> None:

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    label_items = []

    smoothing_window = SMOOTHING_WINDOW_BY_METRIC.get(metric_key, 1)

    for run in runs:
        optim_key = get_optimizer_key(run)
        display_name = get_optimizer_display_name(run)

        steps, values = get_metric_xy(run, metric_key, max_step=max_step)

        if len(values) == 0:
            print(f"Skipping {display_name}: no metric history for {metric_key}")
            continue

        plotted_values = rolling_mean(values, smoothing_window)

        color = OPTIMIZER_COLORS.get(optim_key, None)
        linestyle = OPTIMIZER_LINESTYLES.get(optim_key, "-")

        ax.plot(
            steps,
            plotted_values,
            color=color,
            linestyle=linestyle,
            linewidth=2.3,
        )

        label_items.append({
            "label": display_name,
            "x": steps[-1],
            "y": plotted_values[-1],
            "color": color,
        })

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)

    if metric_key == TRAIN_F1_KEY:
        ax.set_ylim(0.0, 1.05)

    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Extend x-axis slightly to make room for direct labels.
    ax.set_xlim(0, max_step * 1.28)

    label_gap = 0.09 if metric_key == TRAIN_F1_KEY else 0.075

    add_direct_labels(
        ax=ax,
        label_items=label_items,
        x_position=max_step * 1.04,
        min_gap_fraction=label_gap,
    )

    if smoothing_window > 1:
        ax.text(
            0.01,
            0.02,
            f"{smoothing_window}-point moving average",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.70,
        )

    save_figure(filename)


# summary plots

def plot_f1_step_tradeoff(rows: List[Dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for row in rows:
        if row["best_val_f1"] is None or row["step_at_best_val_f1"] is None:
            continue

        optim_key = row["optimizer_key"]
        color = OPTIMIZER_COLORS.get(optim_key, None)

        ax.scatter(
            row["step_at_best_val_f1"],
            row["best_val_f1"],
            s=85,
            color=color,
        )

        ax.annotate(
            row["optimizer"],
            xy=(row["step_at_best_val_f1"], row["best_val_f1"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=10,
            color=color,
        )

    ax.set_title("Validation Performance vs Training Efficiency")
    ax.set_xlabel("Step at Best Validation Macro-F1")
    ax.set_ylabel("Best Validation Macro-F1")

    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    save_figure("final_f1_step_tradeoff.png")


def plot_steps_to_best_f1_horizontal(rows: List[Dict]) -> None:
    valid_rows = [
        row for row in rows
        if row["step_at_best_val_f1"] is not None
        and row["best_val_f1"] is not None
    ]

    valid_rows.sort(key=lambda row: row["best_val_f1"], reverse=True)

    labels = [row["optimizer"] for row in valid_rows]
    steps = [row["step_at_best_val_f1"] for row in valid_rows]
    colors = [OPTIMIZER_COLORS.get(row["optimizer_key"], None) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    bars = ax.barh(labels, steps, color=colors, alpha=0.88)

    ax.invert_yaxis()
    ax.set_title("Step Required to Reach Best Validation Macro-F1")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("")

    ax.xaxis.grid(True, alpha=0.18, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    max_step_value = max(steps)

    for bar, row in zip(bars, valid_rows):
        width = bar.get_width()

        ax.text(
            width + max_step_value * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{row['best_val_f1']:.4f}",
            va="center",
            fontsize=10,
        )

    save_figure("final_steps_to_best_f1_horizontal.png")


def plot_peak_memory_horizontal(rows: List[Dict]) -> None:
    valid_rows = [
        row for row in rows
        if row["peak_gpu_memory_gb"] is not None
    ]

    if len(valid_rows) == 0:
        print("\nWARNING: No GPU memory plot was created.")
        print("No GPU memory metric was found for the selected final runs.")
        print("Check the metric names printed below and update GPU_MEMORY_MB_CANDIDATES if needed.\n")

        for row in rows:
            run = client.get_run(row["run_id"])
            keys = sorted(run.data.metrics.keys())

            print(f"{row['optimizer']} available metric keys:")
            for key in keys:
                print(f"  - {key}")

        return

    valid_rows.sort(key=lambda row: row["peak_gpu_memory_gb"], reverse=True)

    labels = [row["optimizer"] for row in valid_rows]
    values = [row["peak_gpu_memory_gb"] for row in valid_rows]
    colors = [OPTIMIZER_COLORS.get(row["optimizer_key"], None) for row in valid_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    bars = ax.barh(labels, values, color=colors, alpha=0.88)

    ax.invert_yaxis()
    ax.set_title("Peak GPU Memory Usage")
    ax.set_xlabel("Peak GPU Memory Usage (GB)")
    ax.set_ylabel("")

    ax.xaxis.grid(True, alpha=0.18, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    for bar, value in zip(bars, values):
        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f} GB",
            va="center",
            fontsize=10,
        )

    save_figure("final_peak_gpu_memory.png")


# less dense LR sweep plots

def lr_matches_selected(lr: Optional[float], selected_lr: Optional[float]) -> bool:
    if lr is None or selected_lr is None:
        return False

    return abs(lr - selected_lr) <= selected_lr * 1e-6


def make_lr_label(run) -> str:
    lr = get_float_param(run, LR_PARAM_KEY)

    if lr is None:
        return "LR unknown"

    return f"{lr:.2e}"


def plot_lr_sweep_train_loss(optim_key: str, filter_string: str) -> None:
    runs = get_runs(filter_string)

    if len(runs) == 0:
        print(f"No LR sweep runs found for {optim_key}")
        return

    runs = sorted(
        runs,
        key=lambda run: get_float_param(run, LR_PARAM_KEY, default=float("inf")),
    )

    selected_lr = SELECTED_LR_BY_OPTIM.get(optim_key)
    display_name = OPTIMIZER_DISPLAY_NAMES.get(optim_key, optim_key)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    label_items = []

    for run in runs:
        steps, values = get_metric_xy(
            run,
            TRAIN_LOSS_KEY,
            max_step=MAX_LR_SWEEP_STEP,
        )

        if len(values) == 0:
            continue

        lr = get_float_param(run, LR_PARAM_KEY)
        is_selected = lr_matches_selected(lr, selected_lr)

        linewidth = 3.0 if is_selected else 1.5
        alpha = 1.0 if is_selected else 0.55
        zorder = 5 if is_selected else 2

        ax.plot(
            steps,
            values,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

        label = make_lr_label(run)

        if is_selected:
            label = f"{label} selected"

        label_items.append({
            "label": label,
            "x": steps[-1],
            "y": values[-1],
            "color": "black" if is_selected else "dimgray",
        })

    ax.set_title(f"{display_name} Learning-Rate Sweep")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")

    ax.grid(True, alpha=0.18, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_xlim(0, MAX_LR_SWEEP_STEP * 1.20)

    add_direct_labels(
        ax=ax,
        label_items=label_items,
        x_position=MAX_LR_SWEEP_STEP * 1.025,
    )

    filename = f"{optim_key}_lr_sweep_train_loss_clean.png"
    save_figure(filename)


# main execution

def main() -> None:
    # Final comparison
    final_runs_raw = get_runs(FINAL_COMPARISON_FILTER)

    print(f"Final comparison filter returned {len(final_runs_raw)} runs.")

    final_runs = select_final_runs(final_runs_raw)

    print(f"Selected {len(final_runs)} final runs.")

    assert len(final_runs) > 0, (
        "No final comparison runs found. "
        "Check FINAL_COMPARISON_FILTER at the top of the script."
    )

    for run in final_runs:
        optim_name = get_optimizer_display_name(run)
        lr = get_float_param(run, LR_PARAM_KEY)
        wd = get_float_param(run, WEIGHT_DECAY_PARAM_KEY)
        best_step, best_f1 = get_best_metric_point(run, VAL_F1_KEY, max_step=None)

        memory_key = resolve_gpu_memory_metric_key(run)

        print(
            f"{optim_name:18s} | "
            f"lr={lr} | wd={wd} | "
            f"best val F1={best_f1:.4f} at step={best_step} | "
            f"memory key={memory_key}"
        )

    # main final comparison plots
    plot_compact_metric_curves(
        runs=final_runs,
        metric_key=VAL_F1_KEY,
        ylabel="Validation Macro-F1",
        title="Validation Macro-F1 Across Optimizers",
        filename="final_validation_macro_f1_clean.png",
        max_step=MAX_FINAL_STEP,
    )

    plot_compact_metric_curves(
        runs=final_runs,
        metric_key=VAL_LOSS_KEY,
        ylabel="Validation Loss",
        title="Validation Loss Across Optimizers",
        filename="final_validation_loss_clean.png",
        max_step=MAX_FINAL_STEP,
    )

    plot_compact_metric_curves(
        runs=final_runs,
        metric_key=TRAIN_LOSS_KEY,
        ylabel="Training Loss",
        title="Training Loss Across Optimizers",
        filename="final_training_loss_clean.png",
        max_step=MAX_FINAL_STEP,
    )

    plot_compact_metric_curves(
        runs=final_runs,
        metric_key=TRAIN_F1_KEY,
        ylabel="Training Macro-F1",
        title="Training Macro-F1 Across Optimizers",
        filename="final_training_macro_f1_clean.png",
        max_step=MAX_FINAL_STEP,
    )

    # summary plots and CSV
    summary_rows = build_summary_rows(final_runs)

    save_summary_csv(summary_rows)

    plot_f1_step_tradeoff(summary_rows)
    plot_steps_to_best_f1_horizontal(summary_rows)
    plot_peak_memory_horizontal(summary_rows)

    # learning rate sweep plots
    for optim_key, filter_string in LR_SWEEP_FILTERS.items():
        plot_lr_sweep_train_loss(
            optim_key=optim_key,
            filter_string=filter_string,
        )


if __name__ == "__main__":
    main()