import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
import math
import pprint
import axioms
import utils.voting_utils as vu
from collections import Counter
from matplotlib.ticker import ScalarFormatter
import matplotlib

distribution_names = {
    "MALLOWS-RELPHI-R": "Mallows",
    "MALLOWS-0.4": "Mallows",
    "URN-R": "Urn",
    "plackett_luce": "Plackett Luce",
    "single_peaked_conitzer": "Single-Peaked",
    "IC": "Impartial Culture",
    "preflib": "PrefLib"
}
rule_names = {
    'Annealing Score Vector': 'Best Positional Scores',
    'Single Profile Annealing': 'Best Positional Scores',
    'Optimized Scores': 'Best Positional Scores',
    'Anti-Plurality': 'Veto',
    'Plurality Veto': 'Plurality + Veto',
    # 'Borda Min-Max': 'Trimmed Borda',
    'Trimmed Borda': 'Pre-Split Trimmed Borda',
    # 'Kemeny': '4',
    # 'Anti-Plurality': '+',
    # 'Borda': '^',
    # 'Borda Min-Max': '<',
    # 'Dowdall': '3',
    # 'PL MLE': 'd',
    # 'Plurality': 'x',
    # 'Plurality Veto': 'P',
    # 'Random': '*',
    # 'Six Approval': '|',
    # 'Five Approval': '^',
    # 'Four Approval': '<',
    # 'Three Approval': '>',
    # 'Two Approval': '*',
}

fixed_colours = {
    "Optimized Scores": "#b624ff",
    'Anti-Plurality': '#26ef00',
    'Borda': '#00c7ff',
    'Copeland': '#7cf500',
    'Dowdall': '#00fbb1',
    'Five Approval': '#ff0039',
    'Four Approval': '#ff7005',
    'PL MLE': '#ee95f1',
    'Plurality': '#000080',
    'Plurality Veto': '#00278e',
    'Plurality + Veto': '#00278e',
    'Random': '#beb7fe',
    'Six Approval': '#b624ff',
    'Three Approval': '#ffd800',
    'Two Approval': '#d8ff21',
    'Monotonicity': '#2ca02c',  # 17becf
    'Reversal Consistency': '#e377c2',
    'Strong Consistency': '#d62728',
    'Union Consistency': '#1f77b4',
    'Homogeneity': '#d62728'
}
fixed_markers = {
    # Rule markers
    'Kemeny': '+',
    'Anti-Plurality': '^',
    'Veto': '^',
    'Borda': '3',
    'Borda Min-Max': '4',
    'Copeland': '1',
    'Dowdall': '3',
    'PL MLE': 'x',
    'Plurality': '<',
    'Plurality Veto': 'P',
    'Plurality + Veto': 'P',
    'Random': '*',
    'Six Approval': '|',
    'Five Approval': '^',
    'Four Approval': '<',
    'Three Approval': '>',
    'Two Approval': '*',
    'Optimized Scores': '.',
    # Axiom markers
    'Union Consistency': 'x',
    'Reversal Consistency': '^',
    'Monotonicity': '2',
    'Homogeneity': '+',
}

fixed_linestyles = {
    # 'Anti-Plurality': '+',
    # 'Borda': '4',
    # 'Copeland': '1',
    # 'Dowdall': '3',
    # 'PL MLE': 'd',
    # 'Plurality': 'x',
    # 'Plurality Veto': '+',
    # 'Random': '*',
    # 'Six Approval': '|',
    # 'Five Approval': '^',
    # 'Four Approval': '<',
    # 'Three Approval': '>',
    # 'Two Approval': 'v',
    # # Axiom line styles
    'Union Consistency': ':',
    'Reversal Consistency': '--',
    'Monotonicity': '-.',
    'Homogeneity': '--',
}

rule_renaming_map = {
    "Optimized Scores": "Best Positional Scores",
    "Single Profile Annealing": "Best Positional Scores",
    "Anti-Plurality": "Veto",
    "Plurality Veto": "Plurality + Veto",
    "Gold Medals": "Leximax",
    "Empirical": "IRV",
    "All Medals": "Medal Count",
    "F1": "F1 (rule used)",
    "F1_rule-1991": "F1 ('91-'02)",
    "F1_rule-2003": "F1 ('03-'09)",
    "F1_rule-2010": "F1 ('10-'18)",
}
# rule_colour_dict = {
#     'Optimized Scores': [0.89019608, 0.46666667, 0.76078431, 1.],
#     'Veto': [0.83921569, 0.15294118, 0.15686275, 1.],
#     'Borda': [0.12156863, 0.46666667, 0.70588235, 1.],
#     'F1 (rule used)': [0.54901961, 0.3372549, 0.29411765, 1.],
#     'Plurality': [1., 0.49803922, 0.05490196, 1.],
#     'Plurality + Veto': [0.17254902, 0.62745098, 0.17254902, 1.],
#     'Two Approval': [0.58039216, 0.40392157, 0.74117647, 1.],
#     'Empirical': [0.58039216, 0.40392157, 0.74117647, 1.]
# }
# rule_colour_dict = {
#     'Plurality': [0.01719, 0.99787, 0.09494, 1.0],
#     'F1': [1.0, 0.89913, 0.0, 1.0],
#     'Best Positional Scores': [0.0, 0.0, 0.502, 1.0],
#     'Borda': [0.0, 0.29423, 0.21344, 1.0],
#     'Two Approval': [0.84845, 1.0, 0.13263, 1.0],
#     "F1 ('03-'09)": [1.0, 0.27389, 0.00252, 1.0],
#     'Plurality + Veto': [0.40425, 0.82895, 0.0, 1.0],
#     'Empirical': [0.86016, 0.06991, 1.0, 1.0],
#     'Empirical Rule': [0.9961, 0.9725, 0.9961, 1.0],
#     "F1 ('10-'18)": [1.0, 0.0, 0.22527, 1.0],
#     'Borda Min-Max': [0.0, 0.26972, 1.0, 1.0],
#     'Kemeny': [0.0, 0.8757, 1.0, 1.0],
#     'Leximax': [0.95161, 0.68298, 0.95811, 1.0],
#     "F1 ('91-'02)": [1.0, 0.74498, 0.04801, 1.0],
#     'PL MLE': [0.0, 0.9844, 0.69776, 1.0],
#     'Medal Count': [0.78717, 0.36397, 0.96171, 1.0],
#     'Veto': [0.56401, 1.0, 0.08683, 1.0]}
rule_colour_dict = {'Best Positional Scores': [0.0, 0.26972, 1.0, 1.0],
                    'Borda': [0.0, 0.8757, 1.0, 1.0],
                    'Borda Min-Max': [1.0, 0.74498, 0.04801, 1.0],
                    'Empirical': [0.56401, 1.0, 0.08683, 1.0],
                    'IRV': [0.66401, 0.6, 0.38683, 1.0],
                    # 'Empirical Rule': [0.84845, 1.0, 0.13263, 1.0],
                    'F1': [0.0, 0.29423, 0.21344, 1.0],
                    "F1 ('03-'09)": [0.01719, 0.90787, 0.09494, 1.0],
                    "F1 ('10-'18)": [1.0, 0.89913, 0.0, 1.0],
                    "F1 ('91-'02)": [0.96016, 0.26991, 0.1, 1.0],
                    'Kemeny': [1.0, 0.27389, 0.00252, 1.0],
                    'Leximax': [1.0, 0.0, 0.22527, 1.0],
                    'Medal Count': [0.95161, 0.68298, 0.95811, 1.0],
                    'PL MLE': [0.78717, 0.36397, 0.96171, 1.0],
                    'Plurality': [0.0, 0.0, 0.502, 1.0],
                    # 'Plurality + Veto': [0.9961, 0.9725, 0.9961, 1.0],
                    'Plurality + Veto': [0.9961, 0.2725, 0.4961, 1.0],
                    'Two Approval': [0.0, 0.9844, 0.69776, 1.0],
                    'Veto': [0.40425, 0.82895, 0.0, 1.0],
                    'Trimmed Borda': [0.0, 0.31916, 0.1523, 1.0],
                    "Pre-Split Trimmed Borda": [0.0, 0.31916, 0.1523, 1.0],
                    }
rule_marker_dict = {
    "Best Positional Scores": "*",
    "Borda": "P",
    "Borda Min-Max": "3",
    "Kemeny": "x",
    "PL MLE": "2",
    "Plurality": "d",
    "Plurality + Veto": "D",
    "Veto": "s",
    "Two Approval": "^",
    "F1": "o",
    "F1 ('91-'02)": "1",
    "F1 ('03-'09)": "2",
    "F1 ('10-'18)": "3",
    'Empirical Rule': "2",
    'IRV': "o",
    "Medal Count": "o",
    "Leximax": "x",
}
excluded_colours = {
    (0.9961, 0.9725, 0.9961, 0.6),
    (0.9961, 0.9725, 0.9961, 1.0),
    (0.60752, 1.0, 0.1249, 0.6),
    (0.60752, 1.0, 0.1249, 1.0),
    (0.40425, 0.82895, 0.0, 0.6),
    (0.40425, 0.82895, 0.0, 1.0),
(2e-05, 0.99675, 0.12307, 0.6),
(2e-05, 0.99675, 0.12307, 1.0),
(0.53613, 1.0, 0.06242, 0.6),
(0.53613, 1.0, 0.06242, 1.0),
(1.0, 0.93237, 0.0, 0.6),
(1.0, 0.93237, 0.0, 1.0)
}


def get_consistent_color(series_name, colormap='gist_ncar', excluded_colors=None, cache=None, alpha=None,
                         force_alpha=False):
    """
    Get a consistent color for a series name from a colormap, excluding specified colors.

    Parameters:
    -----------
    series_name : str
        The name of the series
    colormap : str or matplotlib.colors.Colormap
        The colormap to select colors from
    excluded_colors : list of tuples
        List of RGB or RGBA colors to exclude
    cache : dict, optional
        Dictionary to use for caching colors. If None, no caching is performed.

    Returns:
    --------
    color : tuple
        RGBA color tuple
    """

    # Initialize excluded colors set
    if excluded_colors:
        excluded = set() if excluded_colors is None else set(tuple(c) for c in excluded_colors)
    else:
        excluded = excluded_colours

    # Check cache first if provided
    if cache is not None and series_name in cache and tuple(cache[series_name]) not in excluded:
        if force_alpha:
            color = cache[series_name]
            color = (color[0], color[1], color[2], alpha)
            cache[series_name] = color
        return cache[series_name]

    # Convert string colormap to colormap object if needed
    cmap = plt.colormaps.get_cmap(colormap) if isinstance(colormap, str) else colormap

    # Try to find a color that's not excluded
    seed = 0
    while True:
        # Get a value between 0 and 1 based on the series name and seed
        hash_input = f"{series_name}_{seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        position = (hash_value % 10000) / 10000.0

        # Get color from colormap
        color = cmap(position)
        color = [round(float(color[0]), 5),
                 round(float(color[1]), 5),
                 round(float(color[2]), 5),
                 round(float(color[3]), 5)]

        if alpha is not None and (cache is None or force_alpha):
            # override default alpha of 1.0
            color = (color[0], color[1], color[2], alpha)

        # Check if color is excluded
        if tuple(color) not in excluded:
            # Cache if requested and return
            if cache is not None:
                cache[series_name] = color
            if alpha and force_alpha:
                color = (color[0], color[1], color[2], alpha)
            print(f"Assigning {series_name} to {color}.")
            return color

        # Try a different seed if this color is excluded
        seed += 1

        # Safety valve to prevent infinite loops
        if seed > 100:
            print(f"Warning: Could not find non-excluded color for '{series_name}' after 100 attempts")
            if cache is not None:
                cache[series_name] = color
            return color


def get_consistent_marker(series_name, marker_list=None, cache=None):
    """
    Get a consistent marker for a series name.

    Parameters:
    -----------
    series_name : str
        The name of the series
    marker_list : list
        List of markers to choose from. If None, uses a default list.
    cache : dict, optional
        Dictionary to use for caching markers. If None, no caching is performed.

    Returns:
    --------
    marker : str
        Matplotlib marker string
    """
    # Check cache first if provided
    if cache is not None and series_name in cache:
        return cache[series_name]

    # Default marker list if none provided
    if marker_list is None:
        marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '+', 'x', "1", "4"]

    # Generate a consistent index based on the series name
    hash_value = int(hashlib.md5(series_name.encode()).hexdigest(), 16)
    marker_idx = hash_value % len(marker_list)

    # Get the marker
    marker = marker_list[marker_idx]

    # Cache if requested
    if cache is not None:
        cache[series_name] = marker

    return marker


def add_series_data_to_axis(ax, data, scatter=False):
    """
    Meant to be useful as a part of making subplots. Make scatter plot of given points on the given axis.
    :param ax:
    :param data:
    :return:
    """
    for series_label, series_data in data["series"].items():
        x_values = series_data["x_values"]
        y_values = series_data["y_values"]
        # colour = _hex_colour_for_label(series_label)
        # line_colour = _hex_to_rgba(h=colour, alpha=0.6)
        line_alpha = 0.6
        # cache = {name: _hex_to_rgba(colour, alpha=line_alpha) for name, colour in fixed_colours.items()}
        line_colour = get_consistent_color(series_name=series_label,
                                           colormap="gist_ncar",
                                           # cache=rule_colour_dict,
                                           cache=rule_colour_dict,
                                           excluded_colors={
                                               (0.97098, 0.80901, 0.97465, 0.6),
                                               (0.0, 0.9853, 0.40946, 0.6),
                                               (0.0, 0.99109, 0.79418, 0.6),
                                               (0.0, 0.77523, 1.0, 0.6),
                                               (0.73245, 1.0, 0.23416, 0.6),
                                               (1.0, 0.0373, 0.0, 0.6)
                                           },
                                           alpha=line_alpha,
                                           force_alpha=True)

        if "marker" in series_data:
            marker = series_data["marker"]
        else:
            marker = "."
        if "linestyle" in series_data:
            linestyle = series_data["linestyle"]
        else:
            linestyle = "solid"

        if scatter:
            ax.scatter(x_values,
                       y_values,
                       label=series_label,
                       marker=marker,
                       # markersize=5,
                       # markerfacecolor=marker_colour,
                       color=line_colour
                       )
            print(f"Plotting {series_label} with colour: {line_colour}")
        else:
            ax.plot(x_values,
                    y_values,
                    label=series_label,
                    marker=marker,
                    # markersize=5,
                    # markerfacecolor=marker_colour,
                    color=line_colour,
                    linestyle=linestyle
                    )


def _hex_colour_for_label(label, cmap="gist_ncar"):
    # Store consistent colours for given axis labels. Useful for consistent colours across multiple plots.

    c = None
    if label in fixed_colours:
        c = fixed_colours[label]
    else:
        # Make "random" colour which is always consistent for a given label
        hash_value = int(hashlib.md5(label.encode('utf-8')).hexdigest()[:8], 16)
        normalized_value = hash_value / (16 ** 8)
        if cmap is not None:
            cmap = plt.get_cmap(cmap)
            r, g, b, a = cmap(normalized_value)
            c = f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}{int(a * 255):02X}"
        else:
            r = abs(hash_value) % 256
            g = abs(hash_value // 256) % 256
            b = abs(hash_value // 65536) % 256
            a = 255
            c = f"#{r:02X}{g:02X}{b:02X}{a:02X}"
    return c


def _hex_to_rgba(h, alpha):
    '''
    Converts color value in hex format to rgba format with alpha transparency.
    '''
    return tuple([int(h.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [alpha])


def _plot_ground_truth_vs_kt_distance(df):
    plot_data = {
        "series": dict(),  # map each series name to dict of data for that series
        "xlabel": "",
        "ylabel": "Violation Rate",
        "title": f"Ground Truth vs KT Distance",
        "series_label": ""
    }

    # Assign a unique color for each voting rule
    voting_rules = df['voting rule'].unique()
    # colors = plt.cm.get_cmap('tab10', len(possible_rules))  # Use a colormap with enough unique colors
    # rule_to_color = {rule: colors(i) for i, rule in enumerate(possible_rules)}

    # Plot the points
    # plt.figure(figsize=(10, 6))
    for rule in voting_rules:
        rule_data = df[df['voting rule'] == rule]
        x = rule_data['Distance from Central Vote']
        y = rule_data['KT Distance Between Splits']

        plot_data["series"][rule] = dict()
        plot_data["series"][rule]["x_values"] = x
        plot_data["series"][rule]["y_values"] = y
        plot_data["series"][rule]["series_label"] = rule
        if rule in fixed_markers:
            plot_data["series"][rule]["marker"] = fixed_markers[rule]

    return plot_data


def organize_legend_handles(ax):
    """
    Give a pyplot plt object, sort the legend labels and remove duplicates.
    :param ax:
    :return:
    """
    handles, labels = ax.get_legend_handles_labels()
    # handles, labels = plot.gca().get_legend_handles_labels()
    labels_to_handles = {l: h for l, h in zip(labels, handles)}
    unique_labels = sorted(set(labels))

    # Put optimized rule(s) first, if they exist
    best_labels = sorted([l for l in unique_labels if "Best" in l])

    irv_labels = sorted([l for l in unique_labels if "IRV" in l])

    # Put "F1" items second, if they exist
    f1_labels = sorted([l for l in unique_labels if "F1" in l])

    # Put Olympic items third, if they exist
    olympic_labels = [l for l in unique_labels if "Leximax" in l] + [l for l in unique_labels if "Medal Count" in l]

    used_labels = best_labels + irv_labels + f1_labels + olympic_labels

    # Put any remaining items last
    remaining_labels = sorted([l for l in unique_labels
                               if l not in used_labels])

    # Combine all categories in the specified order
    ordered_labels = used_labels + remaining_labels

    handles, labels = zip(*[(labels_to_handles[l], l) for l in ordered_labels])

    # handles, labels = zip(*[(labels_to_handles[l], l) for l in unique_labels])

    return handles, labels


def plot_kt_distance_vs_ground_truth_single_plot(show=False, out_folder="plots", out_name="kt_vs_truth.png"):
    """
    Make scatter plot showing KT distance on one axis, distance from some ground truth on another.
    :return:
    """
    # filename = "results/mallows_experiment-teeny.csv"
    filename = "results/pl_experiment.csv"
    df = pd.read_csv(filename)

    fig, ax = plt.subplots(figsize=(6, 4))

    plot_data = _plot_ground_truth_vs_kt_distance(df)
    add_series_data_to_axis(ax, plot_data)

    # Make plot look better
    plt.suptitle(f"Distance from Reference vs Difference Between Splits", fontsize=14)

    ax.set_xlabel('Distance From Reference Ranking', fontsize=12, x=0.5, y=0.09)
    ax.set_ylabel('Distance Between Splits', fontsize=12, x=0.015, y=0.5)

    ax.grid(alpha=0.5)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=4)

    if show:
        plt.show()
    else:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)
        plt.savefig(os.path.join(out_folder, out_name), dpi=300)


def plot_kt_distance_vs_ground_truth_multiple_voter_combos(show=False, out_folder="plots", out_name="kt_vs_truth.png"):
    """
    Make figure with one subplot for each unique number of voters in the results file.
    :param show:
    :param out_folder:
    :param out_name:
    :return:
    """
    filename = "results/experiment-ground_truth_vs_split_distance.csv"
    df = pd.read_csv(filename)

    voter_amounts = df['num_voters'].unique()

    ncols = len(voter_amounts)
    if 2 < ncols < 9:
        ncols = 3
    elif ncols >= 9:
        ncols = 4

    nrows = math.ceil(len(voter_amounts) / ncols)

    fig, axs = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols, sharey="row", sharex="col",
                            constrained_layout=True)

    for rc, ax in np.ndenumerate(axs):
        ax.axis("off")

    for idx, n_voters in enumerate(voter_amounts):
        ax = fig.axes[idx]
        ax.axis("on")
        subdf = df[df["num_voters"] == n_voters]

        plot_data = _plot_ground_truth_vs_kt_distance(subdf)
        add_series_data_to_axis(ax, plot_data)

        ax.set_title(f"{n_voters} Voters")
        ax.grid(alpha=0.5)

        # Set up plot ticks, limits
        ax.set_xlim((-0.05, 0.75))
        x_ticks = [0, 0.2, 0.4, 0.6]
        ax.set_xticks(x_ticks)

        # ax.set_ylim((-0.05, 0.75))
        ax.set_ylim((-0.05, 0.75))
        y_ticks = [0, 0.2, 0.4, 0.6]
        ax.set_yticks(y_ticks)

    # Make plot look better
    plt.suptitle(f"Distance from Reference vs Difference Between Splits", fontsize=14)

    fig.supxlabel('Distance From Reference Ranking', fontsize=12)
    fig.supylabel('Distance Between Splits', fontsize=12)

    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.6, 0.8), ncol=4)

    if show:
        plt.show()
    else:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)
        plt.savefig(os.path.join(out_folder, out_name), dpi=300)


def plot_kt_distance_vs_ground_truth():
    conference = True  # changes formatting of the figure
    # filename = "results/experiment-ground_truth_vs_split_distance-testing-nsplits=10-neurips.csv"
    filename = "results/experiment-ground_truth_vs_split_distance-testing-nsplits=10-trimmed.csv"

    df = pd.read_csv(filename)

    # Define preference distributions to plot
    pref_dists = df['preference distribution'].unique()
    # pref_dists = pref_dists[::-1]

    # Create figure and subplots
    n_cols = max(2, len(pref_dists))
    if conference:
        # fig, axes = plt.subplots(1, n_cols, figsize=(10, 6), sharey=True, sharex=False)
        fig, axes = plt.subplots(1, n_cols, figsize=(14, 5), sharey=False, sharex=False)
    if not conference:
        fig, axes = plt.subplots(1, n_cols, figsize=(14, 6), sharey=True, sharex=False)

    plt.yscale("log")
    plt.xscale("log")

    formatter = ScalarFormatter()
    formatter.set_scientific(False)

    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0

    # Plot each preference distribution, one per axis
    for idx, pref_dist in enumerate(pref_dists):
        plot_data = {
            "series": dict(),  # map each series name to dict of data for that series
            "xlabel": "KT Distance Between Splits",
            "ylabel": "Distance from Central Vote",
            "title": f"Ground Truth vs KT Distance",
            "series_label": ""
        }
        all_points = []

        # Filter data for current preference distribution
        pref_data = df[df['preference distribution'] == pref_dist]

        # Get unique voting rules
        voting_rules = pref_data['voting rule'].unique()

        # Plot data for each voting rule
        for rule in voting_rules:
            rule_data = pref_data[pref_data['voting rule'] == rule]

            # rule_name = rule
            if rule in rule_names:
                rule = rule_names[rule]

            all_x = rule_data['KT Distance Between Splits']
            all_y = rule_data['Distance from Central Vote']
            if rule not in plot_data["series"]:
                plot_data["series"][rule] = dict()

            plot_data["series"][rule]["series_label"] = rule
            plot_data["series"][rule]["x_values"] = []
            plot_data["series"][rule]["y_values"] = []

            for (x, y) in zip(all_x, all_y):
                all_points.append((x, y))
                plot_data["series"][rule]["x_values"].append(x)
                plot_data["series"][rule]["y_values"].append(y)
                plot_data["series"][rule]["marker"] = get_consistent_marker(rule,
                                                                            cache=rule_marker_dict)
                # if rule in fixed_markers:
                #     plot_data["series"][rule]["marker"] = fixed_markers[rule]

        add_series_data_to_axis(ax=axes[idx], data=plot_data, scatter=True)

        # Customize subplot
        subplot_title = pref_dist
        if pref_dist in distribution_names:
            subplot_title = distribution_names[pref_dist]
        axes[idx].set_title(f'{subplot_title}', fontsize=18)
        # axes[idx].legend(loc='upper left')
        axes[idx].grid(True, alpha=0.4)
        axes[idx].tick_params(axis='both', labelsize=16)

        if pref_dist == "MALLOWS-0.4":
            axes[idx].set_xlim((0.025, 0.115))
            axes[idx].set_ylim((0.023, 0.3))
        elif pref_dist == "plackett_luce":
            axes[idx].set_xlim((0.038, 0.13))
            axes[idx].set_ylim((0.055, 0.35))

        # x_ticks = [0.04, 0.07, 0.10, 0.13]
        # axes[idx].set_xticks(x_ticks)
        # axes[idx].set_xticklabels([str(xt) for xt in x_ticks])
        axes[idx].xaxis.set_major_formatter(formatter)
        # axes[idx].xaxis.set_minor_formatter(formatter)

        y_ticks = [0.05, 0.1, 0.2, 0.3]
        axes[idx].set_yticks(y_ticks)
        axes[idx].set_yticklabels([str(yt) for yt in y_ticks])
        axes[idx].yaxis.set_major_formatter(formatter)
        # axes[idx].yaxis.set_minor_formatter(formatter)

        # add line of best fit
        x = np.array([ap[0] for ap in all_points])
        y = np.array([ap[1] for ap in all_points])

        coefficients = np.polyfit(x, y, 1)  # Fit a 1st degree polynomial (linear regression)
        slope, intercept = coefficients

        x = np.linspace(min(x), max(x), 100)
        line_of_best_fit = slope * x + intercept

        label = f"Best Fit: b = {round(intercept, 2)}, m = {round(slope, 2)}"
        axes[idx].plot(x, line_of_best_fit, label=label, color="orange", linestyle="--", alpha=0.5)

        handles, labels = axes[idx].get_legend_handles_labels()
        handles = [handles[-1]]
        labels = [labels[-1]]
        axes[idx].legend(handles, labels, loc="lower right", fontsize=14)

    # replace all ticks on all axes
    for idx in range(len(pref_dists)):
        ax = axes[idx]
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.sca(ax)

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='minor',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,  # ticks along the left edge are off
            right=False,  # ticks along the right edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.minorticks_off()

        ax.xaxis.set_ticks([0.04, 0.08, 0.12])
        ax.yaxis.set_ticks([0.10, 0.20, 0.30])

    handles, labels = organize_legend_handles(axes[0])
    labels = list(labels)
    for idx in range(len(labels)):
        if labels[idx] == "Best Positional Scores":
            labels[idx] = "Best Positional\nScores"
    # handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:], labels[1:]

    # Conference Formatting:
    if conference:
        # # fig.suptitle('Reference Distance vs Split Distance', fontsize=18)
        # fig.supylabel("KT Distance To Ground Truth", fontsize=20, y=0.6)
        # fig.supxlabel("KT Distance Between Splits", fontsize=20, x=0.5, y=0.19)
        # # fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=15)
        # plt.tight_layout(rect=(0, 0, 1, 1))  # l b r t
        # fig.subplots_adjust(bottom=0.3)
        fig.supylabel("KT Distance To Ground Truth", fontsize=20, y=0.55)
        fig.supxlabel("KT Distance Between Splits", fontsize=20, x=0.45, y=0.02)
        # fig.legend(handles, labels, ncols=1, bbox_to_anchor=(0.295, 0.924))
        # fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=15)
        fig.legend(handles, labels, ncols=1, bbox_to_anchor=(0.985, 0.94), fontsize=15)
        # fig.legend(handles, labels, ncols=4, bbox_to_anchor=(0.9, 0.1), fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 1))  # l b r t
        fig.subplots_adjust(right=0.8)

    # ArXiV Formatting:
    if not conference:
        fig.supylabel("KT Distance To Ground Truth", fontsize=20, y=0.55)
        fig.supxlabel("KT Distance Between Splits", fontsize=20, x=0.5, y=0.02)
        # fig.legend(handles, labels, ncols=1, bbox_to_anchor=(0.295, 0.924))
        # fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=15)
        fig.legend(handles, labels, ncols=1, bbox_to_anchor=(1, 0.94), fontsize=15)
        # fig.legend(handles, labels, ncols=4, bbox_to_anchor=(0.9, 0.1), fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 1))  # l b r t
        fig.subplots_adjust(right=0.8)

    plt.show()

    return fig, axes


def plot_kt_distance_vs_ground_truth_multiple_shuffle_amounts(show=False, out_folder="plots",
                                                              out_name="kt_vs_truth-shuffling.png"):
    """
    Make figure with one subplot for each unique number of voters in the results file.
    :param show:
    :param out_folder:
    :param out_name:
    :return:
    """
    filename = "results/experiment-shuffle_top_preferences.csv"
    # filename = "results/experiment-shuffle_top_preferences-kemeny.csv"
    df = pd.read_csv(filename)

    shuffle_amounts = df['Shuffle Amount'].unique()
    shuffle_amounts.sort()

    fig, axs = plt.subplots(figsize=(12, 8), nrows=3, ncols=4, sharey="row", sharex="col", constrained_layout=True)

    for rc, ax in np.ndenumerate(axs):
        ax.axis("off")

    for dist in ["MALLOWS-RELPHI-R", "MALLOWS-RELPHI-0.5", "plackett_luce"]:
        if dist == "MALLOWS-RELPHI-R":
            row = 0
        elif dist == "MALLOWS-RELPHI-0.5":
            row = 1
        elif dist == "plackett_luce":
            row = 2
        else:
            raise ValueError(f"Bad distribution: {dist}")
        for idx, shuffle_amount in enumerate(shuffle_amounts):

            ax = axs[row, idx]
            print(f"Adding to axis ({row}, {idx})")
            # ax = fig.axes[idx]
            ax.axis("on")
            subdf = df[(df["Shuffle Amount"] == shuffle_amount) & (df["preference distribution"] == dist)]
            # subdf = df[df["Shuffle Amount"] == shuffle_amount]

            plot_data = _plot_ground_truth_vs_kt_distance(subdf)
            add_series_data_to_axis(ax, plot_data)

            ax.set_title(f"{shuffle_amount} Top Ranks Shuffle")
            ax.grid(alpha=0.5)

            # Set up plot ticks, limits
            ax.set_xlim((-0.05, 0.55))
            x_ticks = [0, 0.25, 0.5]
            ax.set_xticks(x_ticks)

            # ax.set_ylim((-0.05, 0.75))
            ax.set_ylim((-0.05, 0.55))
            y_ticks = [0, 0.25, 0.5]
            ax.set_yticks(y_ticks)

            if idx == 0:
                ax.set_ylabel(dist)

    # Make plot look better
    plt.suptitle(f"Distance from Reference vs Difference Between Splits", fontsize=14)

    fig.supxlabel('Distance From Reference Ranking', fontsize=12)
    fig.supylabel('Distance Between Splits', fontsize=12)

    plt.tight_layout()

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.6, 0.8), ncol=4)

    if show:
        plt.show()
    else:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)
        plt.savefig(os.path.join(out_folder, out_name), dpi=300)


def _organize_preflib_data_for_plot(df):
    """
    Aggregate the individual elections in preflib data so it is suitable for plotting with other axiom data.
    :param df:
    :return:
    """
    grouped_df = df.groupby(['n_candidates', 'axiom_name']).agg({
        'possible_violations': 'count',
        'total_violations': 'sum',
        'pref_dist': 'first'
    }).reset_index()
    return grouped_df


def plot_axiom_evaluation_results(show=False, n_splits=50, out_folder="plots", out_name="axiom_evaluation_results.png"):
    """
    Make plot with top row showing the violations for each axiom as number of alternatives increases and bottom
    row showing the breakdown of least violating rules for each axiom.
    :param show:
    :param out_folder:
    :param out_name:
    :return:
    """

    # In each subplot in top row, have one series for each number of voters
    filename = "results/axiom_experiment-final.csv"
    df = pd.read_csv(filename)
    # manually update possible violations count for new calculation
    df["possible_violations"] = 500

    preflib_filename = "results/axiom_experiment-preflib.csv"
    preflib_df = pd.read_csv(preflib_filename)
    preflib_df = _organize_preflib_data_for_plot(preflib_df)

    # filter to just have this number of splits in the plot
    df = df[df["n_splits"] == n_splits]

    # merge preflib and other violation data
    df = pd.concat([df, preflib_df])
    # update names of some axioms
    # df = df.replace({'Weak Consistency': 'Union Consistency', 'Reversal Symmetry': 'Reversal Consistency'})
    df = df.replace({'Weak Consistency': 'Union Consistency', 'Reversal Consistency': 'Reversal Symmetry'})

    unique_pref_dists = [
        "IC",
        "URN-R",
        # "MALLOWS-RELPHI-R",
        "MALLOWS-0.4",
        "plackett_luce",
        "single_peaked_conitzer",
        "preflib"
    ]

    axiom_keys = [axioms.weak_consistency.name,
                  # axioms.strong_consistency.name,
                  axioms.reversal_symmetry.name,
                  axioms.monotonicity.name,
                  # axioms.homogeneity.name
                  ]

    all_n_voters = [100]
    # n_elections = 500   # use as maximum number of violations rather than possible_violations
    # all_n_candidates = [5, 10, 20]

    fig, axs = plt.subplots(figsize=(12, 2.1), nrows=len(all_n_voters), ncols=len(unique_pref_dists),
                            sharey="row",
                            sharex="col",
                            constrained_layout=True)

    # for each pref dist and number of voters there is a plot
    # for each plot, there is one series per axiom
    for col, dist in enumerate(unique_pref_dists):
        for row, n_voters in enumerate(all_n_voters):

            ax = axs[col]
            ax.grid(alpha=0.4)

            # get data for each individual series
            for ax_name in axiom_keys:
                sd = dict()
                sd["series"] = get_data_single_axiom_single_num_voters(df,
                                                                       axiom_name=ax_name,
                                                                       num_voters=n_voters,
                                                                       pref_dist=dist)

                add_series_data_to_axis(ax, sd)

            if row == 0:
                dist_name = distribution_names[dist]
                ax.set_title(dist_name)

            # if col == 0:
            #     ax.set_ylabel(f"{n_voters} Voters")

            if row == len(all_n_voters) - 1:
                ax.set_xlim((4, 21))
                x_ticks = [5, 10, 15, 20]
                ax.set_xticks(x_ticks)
                # ax.set_xlabel("Number of Alternatives")

            if col == 0:
                # ax.set_ylim((-0.01, 0.61))
                # y_ticks = [0, 0.2, 0.4, 0.6]
                # ax.set_yticks(y_ticks)

                ax.set_ylim((-0.005, 0.19))
                y_ticks = [0, 0.05, 0.1, 0.15]
                ax.set_yticks(y_ticks)

        # Add a histogram in bottom row for each preference distribution

    # fig.suptitle(f"Axiom Violations")
    # plt.ylim((-0.001, 0.16))
    # plt.yscale("symlog")
    fig.supylabel("Violation Rate")
    fig.supxlabel("Number of Alternatives")
    # ax.legend(ncols=2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, bbox_to_anchor=(1, 0.88))

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder, out_name))


def get_data_single_axiom_single_num_voters(df, axiom_name, num_voters, pref_dist):
    """
    Extract a single series of data showing the axiom violations for a single axiom and number of voters from the
    given axiom violation data file.
    :param df:
    :param axiom_name:
    :param num_voters:
    :param pref_dist:
    :param max_violations:
    :return:
    """
    rows = df[(df["pref_dist"] == pref_dist) & (df["n_voters"] == num_voters) & (df["axiom_name"] == axiom_name)]
    rows = df[(df["pref_dist"] == pref_dist) & (df["axiom_name"] == axiom_name)]

    x_vals = rows["n_candidates"].tolist()
    total_viols = rows["total_violations"].to_numpy()
    possible_viols = rows["possible_violations"].to_numpy()
    # if pref_dist == "preflib":
    # max_violations = max(possible_viols)
    max_violations = possible_viols
    y_vals = np.divide(total_viols, max_violations, out=np.zeros_like(total_viols, dtype=float),
                       where=possible_viols != 0).tolist()
    # y_vals = np.where(y_vals == 0, 1e-3, y_vals)

    series_data = dict()

    series_data[axiom_name] = dict()
    series_data[axiom_name]["x_values"] = x_vals
    series_data[axiom_name]["y_values"] = y_vals
    series_data[axiom_name]["series_label"] = axiom_name
    if axiom_name in fixed_markers:
        series_data[axiom_name]["marker"] = fixed_markers[axiom_name]
    if axiom_name in fixed_linestyles:
        series_data[axiom_name]["linestyle"] = fixed_linestyles[axiom_name]

    return series_data


def get_bar_chart_data(df, pref_dist):
    """
    Get the breakdown of how often each rule was the underlying rule in the rule-picking-rule when the axiom
    was not violated.
    :param df:
    :param pref_dist:
    :return:
    """
    pass


def plot_axiom_histograms(show=False, out_folder="plots", out_name="axiom_rule_histograms.png"):
    figsize = (15, 10)
    filename = "results/axiom_experiment.csv"
    df = pd.read_csv(filename)

    rule_counts = [eval(c) for c in df["non_violating_rule_names"]]
    df["non_violating_rule_names"] = rule_counts
    unique_pref_dists = sorted(df['pref_dist'].unique())

    axiom_keys = [axioms.weak_consistency.name,
                  axioms.strong_consistency.name,
                  axioms.reversal_symmetry.name,
                  axioms.monotonicity.name]
    rule_names = [
        vu.plurality_ranking.name,
        vu.plurality_veto_ranking.name,
        vu.borda_ranking.name,
        vu.antiplurality_ranking.name,
        # vu.copeland_ranking.name,
        # vu.two_approval.name,
        vu.dowdall_ranking.name,
    ]

    # Create subplots - one row per pref_dist, one column per axiom
    fig, axes = plt.subplots(len(unique_pref_dists), len(axiom_keys),
                             figsize=figsize,
                             squeeze=False)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # For each preference distribution (columns)
    for row, pref_dist in enumerate(unique_pref_dists):
        # pref_dist_data = df[df['pref_dist'] == pref_dist]

        # For each axiom
        for col, axiom in enumerate(axiom_keys):
            ax = axes[row, col]

            # Extract violation counts for this axiom from Counter objects
            violation_counts = {}
            counter = df[(df["pref_dist"] == pref_dist) & (df["axiom_name"] == axiom)]['non_violating_rule_names']
            # print(counter)
            if len(counter) == 1:
                counter = counter.tolist()[0]
                # try:
                #     counter = counter[0]
                # except KeyError as kerror:
                #     # happens with an empty Counter
                #     counter = Counter()
            else:
                print(f"Didn't find Counter for {axiom} and {pref_dist}")
                continue
            for rule in rule_names:
                violation_counts[rule] = counter.get(rule, 0)

            print(f"Violation counts: {violation_counts}")

            # normalize violation counts
            vc_sum = sum(violation_counts.values())
            if vc_sum > 0:
                violation_counts = {k: v / sum(violation_counts.values()) for k, v in violation_counts.items()}

            # Create histogram
            bars = ax.bar(list(violation_counts.keys()), list(violation_counts.values()))

            x_ticks = []
            ax.set_xticks([])
            ax.set_xlabel([])

            # Add labels inside bars
            for idx, bar in enumerate(bars):
                # Add value on top half of bar
                height = bar.get_height()
                # ax.text(bar.get_x() + bar.get_width() / 2., height / 2.,
                #         f'{height:,.0f}',
                #         ha='center', va='bottom',
                #         rotation=90)

                # Add category name on bottom half of bar
                ax.text(bar.get_x() + bar.get_width() / 2., height / 4.,
                        list(violation_counts.keys())[idx],
                        ha='center', va='bottom',
                        rotation=90)

            # Set labels and title
            if row == len(unique_pref_dists) - 1:  # Only bottom row gets x-label
                # ax.set_xlabel(f'Violations\nTotal: {pref_dist_data["total_violations"].sum()}')
                ax.set_xlabel(f'Violations\n{axiom}')
            if col == 0:  # Only leftmost column gets y-label
                ax.set_ylabel(f'{pref_dist}')
            # if row == 0:  # Only top row gets title
            #     ax.set_title(axiom)

            # Add grid
            ax.grid(True, alpha=0.3)

    plt.suptitle('Violation Distributions by Preference Distribution and Axiom',
                 y=1.02,
                 fontsize=14)

    if show == True:
        plt.show()
    return fig, axes


def get_colormap_colors(cmap_name, num_colors, hex=False, alpha=None):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name, num_colors)

    colours = []
    # Print colors in order (as RGB values or hex)
    for i in range(num_colors):
        color = cmap(i / (num_colors - 1))  # Normalize i to get colors in order
        # Convert to hex format
        if hex:
            color = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
        else:
            if not alpha:
                alpha = 1.0
            color = [round(float(color[0]), 5),
                     round(float(color[1]), 5),
                     round(float(color[2]), 5),
                     alpha]
        colours.append(color)
    return colours


def print_colormap_with_dict_values(dic, cmap_name, alpha=None):
    colours = get_colormap_colors(cmap_name, num_colors=len(dic), alpha=alpha)

    colour_map = dict(zip(dic.keys(), colours))
    pprint.pprint(colour_map)

    return colour_map


if __name__ == "__main__":
    # rule_marker_dict = {
    #     "Best Positional Scores": "*",
    #     "Borda": "+",
    #     "Borda Min-Max": "3",
    #     "Kemeny": "1",
    #     "PL MLE": "2",
    #     "Plurality": "d",
    #     "Plurality + Veto": "D",
    #     "Veto": "s",
    #     "Two Approval": "^",
    #     "F1": "o",
    #     "F1 ('91-'02)": "1",
    #     "F1 ('03-'09)": "2",
    #     "F1 ('10-'18)": "3",
    #     'Empirical': "2",
    #     "Medal Count": "o",
    #     "Leximax": "x",
    #     "Empirical Rule": "2"
    # }
    # colour_map = print_colormap_with_dict_values(dic=rule_colour_dict,
    #                                              cmap_name="gist_ncar",
    #                                              alpha=1.0)
    # rule_colour_dict = colour_map

    plot_kt_distance_vs_ground_truth()
    # plot_axiom_evaluation_results(show=True)