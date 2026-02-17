import json
import plotly.graph_objects as go

# =============================================================================
# draw_chart.py
# =============================================================================
# Purpose:
# - Plot metrics exported from TensorBoard as JSON (usually via "Download data" or
#   some TB export script).
# - The JSON format assumed here is a list of rows: [wall_time, step, value]
#   e.g.:
#     [
#       [1700000000.0, 0, 0.12],
#       [1700000001.0, 512, 0.11],
#       ...
#     ]
#
# Output:
# - Saves a static PNG using Plotly's image export.
#   NOTE: Plotly static export typically requires the "kaleido" package installed:
#     pip install -U kaleido
#
# This file is NOT used by training or evaluation; it is just a utility script.
# =============================================================================


# =============================================================================
# Select which exported TensorBoard JSON files to compare
# =============================================================================
# You can switch metric types by changing the filenames below.
#
# Examples you used before (commented):
# - rollout_ep_len_mean
# - stats_collision_steps_rate
# - stats_unique_reaches
# =============================================================================

# file_paths = [
#     "run-models_tb_logs_PPO_1-tag-rollout_ep_len_mean (1).json",
#     "run-models_tb_logs_PPO_5-tag-rollout_ep_len_mean (2).json"
# ]

# file_paths = [
#     "run-models_tb_logs_PPO_1-tag-stats_collision_steps_rate.json",
#     "run-models_tb_logs_PPO_5-tag-stats_collision_steps_rate.json"
# ]

# file_paths = [
#     "run-models_tb_logs_PPO_1-tag-stats_unique_reaches.json",
#     "run-models_tb_logs_PPO_5-tag-stats_unique_reaches.json"
# ]

file_paths = [
    "run-models_tb_logs_PPO_3-tag-stats_collision_steps_rate.json",
    "run-models_tb_logs_PPO_5-tag-stats_collision_steps_rate (1).json"
]


# =============================================================================
# Data loading / smoothing helpers
# =============================================================================
def load_trace(path: str):
    """
    Load TensorBoard-exported JSON and return (steps, values).

    Expected JSON format:
      list of [wall_time, step, value]
    """
    with open(path, "r") as f:
        data = json.load(f)

    steps = [row[1] for row in data]
    values = [row[2] for row in data]
    return steps, values


def smooth(values, alpha: float = 0.8):
    """
    Apply exponential moving average (EMA) smoothing.

    Matches TensorBoard's smoothing slider behavior approximately:
      alpha=0.8 means "heavy smoothing" (closer to the previous smoothed value).
    """
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


# =============================================================================
# Build Plotly figure
# =============================================================================
fig = go.Figure()

for idx, path in enumerate(file_paths):
    # Extract a short id from filename:
    # e.g. "run-...PPO_3-tag-..." -> "3"
    try:
        name = path.split("PPO_")[1].split("-tag")[0]
    except Exception:
        name = f"trace_{idx}"

    # Load + smooth
    steps, raw_vals = load_trace(path)
    y_sm = smooth(raw_vals, alpha=0.8)  # same as TensorBoard “smoothing=0.8”

    # Your original logic: first line is "No LiDAR", second is "LiDAR included"
    # (purely label choice; doesn't affect data)
    line_name = "No LiDAR" if idx == 0 else "LiDAR included"

    # Keep the same structure you had (explicit list build)
    steps_f = []
    y_f = []
    for s, y in zip(steps, y_sm):
        steps_f.append(s)
        y_f.append(y)

    fig.add_trace(
        go.Scatter(
            x=steps_f,
            y=y_f,
            mode="lines",
            name=line_name,
        )
    )

# =============================================================================
# Layout / titles
# =============================================================================
fig.update_layout(
    title="Average collisions per step",
    xaxis_title="Training Step",
    yaxis_title="Collisions per step",
    # If you switch metrics, update the labels above (optional).
    template="plotly_white"
)

# =============================================================================
# Save image
# =============================================================================
# Uncomment the output you want. Current output kept as-is:
# fig.write_image("ep_len.png", scale=3)
# fig.write_image("coll_per_step.png", scale=3)
# fig.write_image("reaches_per_rollout.png", scale=3)
fig.write_image("lidar_coll.png", scale=3)
