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
        'Annealing Score Vector': 'Optimized Scores',
        'Anti-Plurality': 'Veto'
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
    'Random': '#beb7fe',
    'Six Approval': '#b624ff',
    'Three Approval': '#ffd800',
    'Two Approval': '#d8ff21',
    'Monotonicity': '#2ca02c', #17becf
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
# linestyle_tuple = [
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 5))),
#      ('densely dotted',        (0, (1, 1))),
#
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),
#
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


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
        colour = _hex_colour_for_label(series_label)
        line_colour = _hex_to_rgba(h=colour, alpha=0.6)
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
    filename = "results-final/experiment-ground_truth_vs_split_distance-testing-nsplits=10.csv"
    df = pd.read_csv(filename)

    # Define preference distributions to plot
    pref_dists = df['preference distribution'].unique()
    # pref_dists = pref_dists[::-1]

    # Create figure and subplots
    n_cols = max(2, len(pref_dists))
    fig, axes = plt.subplots(1, n_cols, figsize=(10, 6), sharey=True, sharex=False)

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
                if rule in fixed_markers:
                    plot_data["series"][rule]["marker"] = fixed_markers[rule]

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
            axes[idx].set_xlim((0.03, 0.13))
        elif pref_dist == "plackett_luce":
            axes[idx].set_xlim((0.039, 0.13))

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
            axis='x',  # changes apply to the x-axis
            which='minor',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        ax.xaxis.set_ticks([0.04, 0.08, 0.12])
        ax.yaxis.set_ticks([0.05, 0.10, 0.20, 0.30])

    # fig.suptitle('Reference Distance vs Split Distance', fontsize=18)
    fig.supylabel("KT Distance To Ground Truth", fontsize=20, y=0.6)
    fig.supxlabel("KT Distance Between Splits", fontsize=20, x=0.5, y=0.19)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[:-1], labels[:-1]
    # fig.legend(handles, labels, ncols=1, bbox_to_anchor=(0.295, 0.924))
    fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=15)
    # fig.legend(handles, labels, ncols=1, bbox_to_anchor=(0.334, 0.924), fontsize=12)
    # fig.legend(handles, labels, ncols=4, bbox_to_anchor=(0.9, 0.1), fontsize=12)

    plt.tight_layout(rect=(0, 0, 1, 1))  # l b r t
    fig.subplots_adjust(bottom=0.3)
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


def plot_preflib_scatter(show=True, out_folder="plots", out_name="axiom_evaluation_results-preflib.png"):
    """
    Make scatterplots for each axiom showing whether each election violates or does not violate the axiom.
    One row for each num_splits.
    :param show:
    :param out_folder:
    :param out_name:
    :return:
    """

    # In each subplot in top row, have one series for each number of voters
    filename = "results/axiom_experiment-preflib.csv"
    df = pd.read_csv(filename)

    df = df[df["n_voters"] != "all"]
    df = df[df["n_candidates"] != "all"]

    # Get unique axiom names for subplots
    unique_axioms = df['axiom_name'].unique()
    n_axioms = len(unique_axioms)

    df['n_voters'] = df['n_voters'].apply(int)
    df['n_candidates'] = df['n_candidates'].apply(int)

    max_num_voters = max(df["n_voters"])
    max_num_candidates = max(df["n_candidates"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, n_axioms, figsize=(5 * n_axioms, 8))

    # plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Function to assign colors based on total_violations
    def get_colour(data):
        return ['red' if v == 1 else 'green' for v in data['total_violations']]

    def get_alpha(data):
        return [0.3 if v == 1 else 0.15 for v in data['total_violations']]

    # Create subplots for n_splits = 20
    for idx, axiom in enumerate(unique_axioms):
        filtered_data = df[(df['axiom_name'] == axiom) & (df['n_splits'] == 20)]
        colour = get_colour(filtered_data)
        alpha = get_alpha(filtered_data)
        axes[0, idx].scatter(filtered_data['n_voters'], filtered_data['n_candidates'], c=colour, marker=".",
                             alpha=alpha)
        axes[0, idx].set_title(f'{axiom}')
        # axes[0, idx].set_xlabel('Number of Voters')
        # axes[0, idx].set_ylabel('Number of Candidates')
        # axes[0, idx].grid(True)

    # Create subplots for n_splits = 40
    for idx, axiom in enumerate(unique_axioms):
        filtered_data = df[(df['axiom_name'] == axiom) & (df['n_splits'] == 40)]
        colour = get_colour(filtered_data)
        alpha = get_alpha(filtered_data)
        axes[1, idx].scatter(filtered_data['n_voters'], filtered_data['n_candidates'], c=colour, marker=".",
                             alpha=alpha)
        # axes[1, idx].set_title(f'{axiom}\n(n_splits=40)')
        # axes[1, idx].set_xlabel('Number of Voters')
        # axes[1, idx].set_ylabel('Number of Candidates')
        # axes[1, idx].grid(True)

    for coord, ax in np.ndenumerate(axes):
        ax.set_xlim((-1, max_num_voters + 1))
        ax.set_xticks([x for x in range(0, max_num_voters + 9, 10)])

        ax.set_ylim((4, max_num_candidates + 1))
        ax.set_yticks([y for y in range(5, max_num_candidates + 1, 5)])

        ax.grid(alpha=0.15)

    axes[0, 0].set_ylabel("20 splits")
    axes[1, 0].set_ylabel("40 splits")

    fig.supylabel("Number of Candidates")
    fig.supxlabel("Number of Voters")
    fig.suptitle("Axiom Violations on Preflib Data")
    plt.tight_layout()

    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='red', markersize=10, label='Violation'),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='green', markersize=10, label='No Violation')]
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))

    # plt.suptitle('Voters vs Candidates by Axiom and Number of Splits', fontsize=16, y=1.05)

    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_folder, out_name))


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


def plot_axiom_evaluation_results(show=False, n_splits=20, out_folder="plots", out_name="axiom_evaluation_results.png"):
    """
    Make plot with top row showing the violations for each axiom as number of alternatives increases and bottom
    row showing the breakdown of least violating rules for each axiom.
    :param show:
    :param out_folder:
    :param out_name:
    :return:
    """

    # In each subplot in top row, have one series for each number of voters
    filename = "results-final/axiom_experiment-final.csv"
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
    df = df.replace({'Weak Consistency': 'Union Consistency', 'Reversal Symmetry': 'Reversal Consistency'})

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


def get_colormap_colors(cmap_name, num_colors):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name, num_colors)

    colours = []
    # Print colors in order (as RGB values or hex)
    for i in range(num_colors):
        color = cmap(i / (num_colors - 1))  # Normalize i to get colors in order
        # Convert to hex format
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )
        # print(hex_color)
        colours.append(hex_color)
    return colours


def print_colormap_with_dict_values(dic, cmap_name):
    colours = get_colormap_colors(cmap_name, len(dic))

    colour_map = dict(zip(dic.values(), colours))
    pprint.pprint(colour_map)


if __name__ == "__main__":
    plot_kt_distance_vs_ground_truth()
    # plot_preflib_scatter(show=False)

    n_splits = 50
    out_name = f"axiom_evaluation_results-n_splits={n_splits}.png"
    plot_axiom_evaluation_results(show=False, n_splits=n_splits, out_name=out_name)

    # plot_axiom_histograms(show=True)
    # axdict = [axioms.weak_consistency.name,
    #               axioms.strong_consistency.name,
    #               axioms.reversal_symmetry.name,
    #               axioms.monotonicity.name,
    #           axioms.homogeneity.name]
    # axdict = {n:n for n in axdict}
    # print_colormap_with_dict_values(axdict, "tab10")

    # plot_kt_distance_vs_ground_truth_single_plot(show=True)
    # plot_kt_distance_vs_ground_truth_multiple_voter_combos(show=True)
    # plot_kt_distance_vs_ground_truth_multiple_shuffle_amounts(show=True)
