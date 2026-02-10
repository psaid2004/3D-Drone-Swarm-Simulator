import json

import plotly.graph_objects as go

'''File for plotting & Saving Tensorflow charts'''
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


def load_trace(path):
    """Load JSON list of [time, step, value] and return steps & values."""
    data = json.load(open(path, "r"))
    steps = [row[1] for row in data]
    values = [row[2] for row in data]
    return steps, values


def smooth(values, alpha=0.8):
    """Apply EMA smoothing with weight alpha."""
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


# Build the Plotly figure
fig = go.Figure()

for path in file_paths:
    name = path.split("PPO_")[1].split("-tag")[0]
    steps, raw_vals = load_trace(path)
    y_sm = smooth(raw_vals, alpha=0.8)  # same as TensorBoard “smoothing=0.8”
    line_name = "No LiDAR" if path == file_paths[0] else "LiDAR included"
    steps_f = []
    y_f = []
    for s, y in zip(steps, y_sm):
        steps_f.append(s)
        y_f.append(y)
    fig.add_trace(go.Scatter(
        x=steps_f, y=y_f, mode="lines",
        name=line_name,
    ))

fig.update_layout(
    title="Average collisions per step",
    xaxis_title="Training Step",
    yaxis_title="Collisions per step",
    # title="# Drones reaching their goals",
    # xaxis_title="Training Step",
    # yaxis_title="# Drones Reached",
    template="plotly_white"
)

# write out
# fig.write_image("ep_len.png", scale=3)
# fig.write_image("coll_per_step.png", scale=3)
# fig.write_image("reaches_per_rollout.png", scale=3)
fig.write_image("lidar_coll.png", scale=3)
