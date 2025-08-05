import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re
import numpy as np
from collections import Counter
import plotting as plt_util
from plotting import rule_renaming_map, rule_colour_dict, rule_marker_dict


def scatter_plot_olympics():
    # Load the CSV file
    df = pd.read_csv('results/olympic_data-neurips.csv')

    # df = df.replace('Plurality Veto', 'Plurality + Veto')
    # df = df.replace('Single Profile Annealing', 'Optimized Scores')
    df = df[df["rule_name"] != "Three Approval"]

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    # Extract year from the "Game" column using regular expressions
    df['Year'] = df['Game'].apply(lambda x: int(re.search(r'(\d{4})', x).group(1)))

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
        'Borda',
        'Plurality',
        'Two Approval',
        'Leximax',
        'Medal Count'
    ]

    # Get unique rule_names for coloring
    unique_rules = df['rule_name'].unique()

    # Create a scatter plot
    plt.figure(figsize=(14, 6))

    # Plot each rule_name with different colors and markers
    for rule in unique_rules:
        if rule not in rules_to_display:
            continue
        mask = df['rule_name'] == rule
        colour = plt_util.get_consistent_color(rule,
                                               cache=rule_colour_dict)
        # colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        marker = rule_marker_dict[rule] if rule in rule_marker_dict else plt_util.get_consistent_marker(rule)
        plt.scatter(
            df.loc[mask, 'Year'],
            df.loc[mask, 'distance'],
            color=colour,
            marker=marker,
            s=40,
            label=rule,
            alpha=0.7
        )

    # plt.xlim((0.015, 0.145))

    # # sort legend entries
    # lines = plt.gca().get_lines()
    # sorted_lines = sorted(lines, key=lambda line: np.max(line.get_ydata()))
    # plt.legend(handles=sorted_lines, loc="lower right")
    # plt.rc('xtick', labelsize=30)  # fontsize of the x and y labels
    plt.gca().tick_params(axis='both', which='major', labelsize=12)

    # Add labels and title
    plt.ylabel('Distance', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    # plt.title('Split Distance of Olympic Medals', fontsize=18)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    handles, labels = plt_util.organize_legend_handles(plt.gca())
    plt.legend(handles, labels, ncols=1, bbox_to_anchor=(1, 1), fontsize=15)
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("preflib/plots/olympics_scatter.png")


def bar_plot_olympics():
    df = pd.read_csv('results/olympic_data-neurips.csv')

    # df = df.replace('Plurality Veto', 'Plurality + Veto')
    # df = df.replace('Single Profile Annealing', 'Optimized Scores')
    df = df[df["rule_name"] != "Three Approval"]

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    rules_to_display = [
        'Best Positional Scores',
        # 'Veto',
        'Borda',
        'Plurality',
        'Two Approval',
        'Leximax',
        'Medal Count'
    ]

    # df = df[df['Dataset'].str.contains('F1')]
    # df['rule_name'] = df['rule_name'].replace(regex=r'F1-\d{4}', value='F1')
    # mean_distances = df.groupby('rule_name')['distance'].mean().reset_index()
    mean_distances = df.groupby('rule_name')['distance'].agg(['mean', 'sem']).reset_index()
    mean_distances['sem'] = mean_distances['sem'].fillna(0)
    mean_distances = mean_distances[mean_distances['rule_name'].isin(rules_to_display)]

    # Extract year from the "Game" column using regular expressions
    df['Year'] = df['Game'].apply(lambda x: int(re.search(r'(\d{4})', x).group(1)))

    # Sort specific rules by increasing distance
    mean_distances = mean_distances.sort_values('mean')
    color_dict = {
        # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        rule: plt_util.get_consistent_color(rule,
                                            cache=rule_colour_dict)
        for rule in mean_distances['rule_name']
    }
    colors = [c for rule, c in color_dict.items()]

    # Update annealing rule name to fit better
    mean_distances['rule_name'] = mean_distances['rule_name'].apply(lambda x: x if x != "Best Positional Scores" else "Best Positional\nScores")

    plt.figure(figsize=(10, 4.5))
    plt.grid(True, alpha=0.3, axis="y")
    bars = plt.bar(
        mean_distances['rule_name'], mean_distances['mean'],
        yerr=mean_distances["sem"],
        color=colors,
        # error_kw={'elinewidth': 1.5, 'alpha': 0.5}
    )

    # plt.title('Split Distance on Olympic Events', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Rule Name", fontsize=20)
    plt.ylabel("Distance", fontsize=20)
    plt.ylim((0.24, 0.38))
    plt.gca().tick_params(axis='both', which='major', labelsize=15)

    # Add the actual average values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.03 * max(mean_distances['mean']),
                 f'{height:.3f}', ha='center', va='bottom', size=13)

    plt.tight_layout()
    # plt.show()
    plt.savefig("preflib/plots/olympics_bar.png")


def scatter_plot_f1():
    # Load the CSV file
    df = pd.read_csv('preflib/analysis_results-neurips.csv')

    # df = df.replace('Plurality Veto', 'Plurality + Veto')
    # df = df.replace('Single Profile Annealing', 'Optimized Scores')

    df = df[df['Dataset'].str.contains('F1')]

    # Extract year from the "Game" column using regular expressions
    df['Year'] = df['Dataset'].apply(lambda x: int(re.search(r'(\d{4})', x).group(1)))
    df['rule_name'] = df['rule_name'].replace(regex=r'F1-\d{4}', value='F1')

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    # Get unique rule_names for coloring
    unique_rules = df['rule_name'].unique()

    # Create a scatter plot
    plt.figure(figsize=(14, 6))

    # Plot each rule_name with different colors and markers
    for rule in unique_rules:
        mask = df['rule_name'] == rule
        # colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        colour = plt_util.get_consistent_color(rule,
                                               cache=rule_colour_dict)
        marker = rule_marker_dict[rule] if rule in rule_marker_dict else plt_util.get_consistent_marker(rule)
        plt.scatter(
            df.loc[mask, 'Year'],
            df.loc[mask, 'distance'],
            color=colour,
            marker=marker,
            s=40,
            label=rule,
            # alpha=0.7
        )

    # plt.xlim((0.015, 0.145))
    plt.ylim((0.04, 0.54))

    # Add labels and title
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel('Distance', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    # plt.title('Split Distance of F1 Races', fontsize=18)

    # handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = plt_util.organize_legend_handles(plt.gca())
    plt.legend(handles, labels, ncols=1, bbox_to_anchor=(1, 1.024), fontsize=15)
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("preflib/plots/F1_scatter.png")


def bar_plot_f1():
    df = pd.read_csv('preflib/analysis_results-neurips.csv')

    df = df[df['Dataset'].str.contains('F1')]
    df['rule_name'] = df['rule_name'].replace(regex=r'F1-\d{4}', value='F1')

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    # Extract year from the "Game" column using regular expressions
    df['Year'] = df['Dataset'].apply(lambda x: int(re.search(r'(\d{4})', x).group(1)))

    filter_f1_by_active_year = False
    if filter_f1_by_active_year:
        non_f1_mask = ~df['rule_name'].str.contains('F1', na=False)
        f1_special_mask = (
                df['rule_name'].str.contains('F1', na=False) &
                (
                        ((df['Year'] >= 2010) &
                         (df['Year'] <= 2018) &
                         (df['rule_name'] == "F1 ('10-'18)")) |
                        ((df['Year'] >= 1991) &
                         (df['Year'] <= 2002) &
                         (df['rule_name'] == "F1 ('91-'02)")) |
                        ((df['Year'] >= 2003) &
                         (df['Year'] <= 2009) &
                         (df['rule_name'] == "F1 ('03-'09)"))
                )
        )
        # optimized_mask = (
        #         df['rule_name'].str.contains('Best', na=False) &
        #         (
        #                 (df['Year'] >= 2010) &
        #                  (df['Year'] <= 2018)
        #         )
        # )
        final_mask = non_f1_mask | f1_special_mask
        # final_mask = f1_special_mask | optimized_mask
        # Apply the mask to filter the dataframe
        df = df[final_mask]

    # add column to label which set of years each row is in
    bins = [1990, 2002, 2009, 2018]
    period_labels = ["'91-'02", "'03-'09", "'10-'18"]
    df["period"] = pd.cut(df["Year"], bins=bins, labels=period_labels, include_lowest=True)

    # mean_distances = df.groupby('rule_name')['distance'].mean().reset_index()
    mean_distances = df.groupby(['period', 'rule_name'])['distance'].agg(['mean', 'sem']).reset_index()

    # mean_distances = df.groupby('rule_name')['distance'].agg(['mean', 'sem']).reset_index()
    mean_distances['sem'] = mean_distances['sem'].fillna(0)

    ####################
    # START NEW CODE
    ####################

    pivot_data = mean_distances.pivot_table(index='period', columns='rule_name', values='mean', aggfunc='mean')
    pivot_data['F1'] = pivot_data["F1 ('10-'18)"].fillna(pivot_data["F1 ('91-'02)"]).fillna(pivot_data["F1 ('03-'09)"])

    if filter_f1_by_active_year:
        pivot_data = pivot_data.drop(["F1 ('10-'18)", "F1 ('91-'02)", "F1 ('03-'09)"], axis=1)

    pivot_sem = mean_distances.pivot_table(index='period', columns='rule_name', values='sem', aggfunc='mean')
    temp_data = pivot_sem[["F1 ('10-'18)", "F1 ('91-'02)", "F1 ('03-'09)"]].replace(0, np.nan)
    pivot_sem['F1'] = temp_data["F1 ('10-'18)"].fillna(temp_data["F1 ('91-'02)"]).fillna(temp_data["F1 ('03-'09)"])
    if filter_f1_by_active_year:
        pivot_sem = pivot_sem.drop(["F1 ('10-'18)", "F1 ('91-'02)", "F1 ('03-'09)"], axis=1)

    if filter_f1_by_active_year:
        rule_order = ["F1", "Best Positional Scores", "Borda", "Two Approval", "Plurality + Veto", "Plurality", "Veto"]
    else:
        rule_order = ["F1 ('91-'02)", "F1 ('03-'09)", "F1 ('10-'18)", "Best Positional Scores", "Borda", "Two Approval", "Plurality + Veto", "Plurality", "Veto"]
    pivot_data = pivot_data.reindex(columns=rule_order)
    pivot_sem = pivot_sem.reindex(columns=rule_order)

    # Get the periods and rule names
    periods = pivot_data.index
    rule_names = pivot_data.columns
    n_periods = len(periods)
    n_rules = len(rule_names)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Set the width of bars and positions
    bar_width = 0.9 / n_rules  # Adjust width based on number of rule names

    color_dict = {
        # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        rule: plt_util.get_consistent_color(rule,
                                            cache=rule_colour_dict)
        for rule in pivot_data.columns
    }

    all_bars = []
    # Create bars for each rule_name
    for i, period in enumerate(periods):
        positions = []
        values = []
        errors = []
        colors = []
        # rule_names = []
        for j, rule_name in enumerate(rule_names):
            # Get values for this rule_name, handling missing combinations
            # values = [pivot_data.loc[period, rule_name] if not pd.isna(pivot_data.loc[period, rule_name])
            #           else 0 for period in periods]
            val = pivot_data.loc[period, rule_name] if not pd.isna(pivot_data.loc[period, rule_name]) else None
            err = pivot_sem.loc[period, rule_name] if not pd.isna(pivot_sem.loc[period, rule_name]) else None
            if val is None:
                continue

            # values = [pivot_data.loc[period, rule_name] for rule_name in rule_names if not pd.isna(pivot_data.loc[period, rule_name])]

            # Calculate position for this set of bars
            # positions = x + (i - n_rules / 2 + 0.5) * bar_width
            position = i + (j - n_rules/2)*bar_width
            values.append(val)
            errors.append(err)
            positions.append(position)
            # rule_names.append(rule_name)
            colors.append(color_dict[rule_name])

        # Create the bars
        bars = ax.bar(positions, values, bar_width, color=colors, yerr=errors, label=period, alpha=0.8)
        all_bars += bars

        rule_name_labels = [
            rule_names[idx] if rule_names[idx] != "Best Positional Scores" else "Best Positional\nScores" for idx in
            range(len(rule_names))]
        # Add labels underneath each bar for rule names
        for j, (pos, val) in enumerate(zip(positions, values)):
            # if val > 0:  # Only add label if there's actually a bar
            # You can customize what text to show - here showing the rule_name
            plt.text(pos, 0.072, rule_name_labels[j],
                    ha='center', va='top', rotation=85, fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.xticks([], [])
    # plt.xlabel("Rule Name", fontsize=20)
    plt.ylabel("Distance", fontsize=16)
    plt.ylim((0.08, 0.54))
    plt.gca().tick_params(axis='both', which='major', labelsize=16)

    for period_idx in [1, 2, 3]:
        plt.text(x=period_idx-1, y=0.52, s=f"F1 {period_labels[period_idx - 1]}",
                 ha='center', va='top', rotation=0,
                 # fontweight="bold",
                 fontsize=16
                 )

    # Add the actual average values on top of each bar
    for bar in all_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.03 * max(mean_distances['mean']),
                 f'{height:.2f}', ha='center', va='bottom', size=10)

    # plt.subplots_adjust(bottom=0.5)
    plt.tight_layout(rect=[0, 0.0, 1, 1])

    # plt.show()
    plt.savefig("preflib/plots/F1_bar.png")

    ####################
    # END NEW CODE
    ####################

    # # Separate optimized rule, F1 rules and specific rules
    # optimized_rule = mean_distances[mean_distances['rule_name'].str.contains('Best')]
    # f1_rules = mean_distances[mean_distances['rule_name'].str.contains('F1')]
    # # other_rules = mean_distances[~mean_distances['rule_name'].str.contains('F1')]
    # other_rules = mean_distances[
    #     ~(mean_distances['rule_name'].str.contains('F1') | mean_distances['rule_name'].str.contains('Best'))]
    #
    # # Sort specific rules by increasing distance
    # f1_rules = f1_rules.sort_values("rule_name")
    #
    # rule_order = ["F1 ('91-'02)", "F1 ('03-'09)", "F1 ('10-'18)"]
    # f1_rules['rule_name'] = pd.Categorical(f1_rules['rule_name'], rule_order)
    # f1_rules = f1_rules.sort_values('rule_name')
    # # specific_rules_data = other_rules.sort_values('rule_name')
    # specific_rules_data = other_rules.sort_values('mean')
    # # specific_rules_data = other_rules.sort_
    #
    # # Combine back with F1 rules first
    # if not other_rules.empty:
    #     mean_distances = pd.concat([f1_rules, optimized_rule, specific_rules_data]).reset_index(drop=True)
    # else:
    #     mean_distances = specific_rules_data
    #
    # color_dict = {
    #     # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
    #     rule: plt_util.get_consistent_color(rule,
    #                                         cache=rule_colour_dict)
    #     for rule in mean_distances['rule_name']
    # }
    # colors = [c for rule, c in color_dict.items()]
    #
    # # Update annealing rule name to fit better
    # mean_distances['rule_name'] = mean_distances['rule_name'].apply(lambda x: x if x != "Best Positional Scores" else "Best Positional\nScores")
    #
    # plt.figure(figsize=(10, 4.5))
    # plt.grid(True, alpha=0.3, axis="y")
    # bars = plt.bar(
    #     mean_distances['rule_name'], mean_distances['mean'],
    #     yerr=mean_distances["sem"],
    #     color=colors,
    #     # error_kw={'elinewidth': 1.5, 'alpha': 0.5}
    # )
    #
    # # plt.title('Split Distance on F1 Races', fontsize=18)
    # plt.xticks(rotation=45, ha='right')
    # # plt.xlabel("Rule Name", fontsize=20)
    # plt.ylabel("Distance", fontsize=26)
    # plt.ylim((0.08, 0.54))
    # plt.gca().tick_params(axis='both', which='major', labelsize=20)
    #
    # # Add the actual average values on top of each bar
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height + 0.03 * max(mean_distances['mean']),
    #              f'{height:.2f}', ha='center', va='bottom', size=18)
    #
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("preflib/plots/F1_bar.png")


def scatter_plot_preflib(include_zero_valued_elections=True, exclude_elections_above_max_y=None, show_num_voters_and_cands=False):
    """

    :param include_zero_valued_elections: If False, do not include elections where every rule has a split distance of 0.
    :param exclude_elections_above_max_y: If set to a value, do not include any elections where any rule has a max split
    distance greater than the given value.
    :return:
    """
    # Load the CSV file
    df = pd.read_csv('preflib/analysis_results-neurips.csv')

    df = df.replace('Plurality Veto', 'Plurality + Veto')
    df = df.replace('Single Profile Annealing', 'Optimized Scores')

    # Filter to the values NOT containing F1 data
    df = df[~df['Dataset'].str.contains('F1')]
    df = df[~df['rule_name'].str.contains('F1')]

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    # # Extract year from the "Game" column using regular expressions
    # df['Year'] = df['Dataset'].apply(lambda x: int(re.search(r'(\d{4})', x).group(1)))
    # df['rule_name'] = df['rule_name'].replace(regex=r'City*', value='Empirical')
    df['rule_name'] = df['rule_name'].apply(lambda x: "IRV" if "City" in str(x) else x)
    df['rule_name'] = df['rule_name'].apply(lambda x: "IRV" if "UK Labour" in str(x) else x)

    elections_to_remove = []
    for election in df["Dataset"].unique():
        max_distance = df[df["Dataset"] == election]["distance"].max()
        min_distance = df[df["Dataset"] == election]["distance"].min()

        if not include_zero_valued_elections and (max_distance == min_distance == 0):
            elections_to_remove.append(election)
        if exclude_elections_above_max_y and max_distance > exclude_elections_above_max_y:
            elections_to_remove.append(election)
    df = df[~df['Dataset'].isin(elections_to_remove)]

    mean_distances = df.groupby(['Dataset', 'rule_name', 'n_alternatives', 'n_voters'])['distance'].mean().reset_index()
    mean_distances = mean_distances.sort_values('Dataset')

    city_order = ["Burlington",
                  "Aspen",
                  "Berkeley",
                  "Minneapolis",
                  "Oakland",
                  "Pierce",
                  "San Francisco",
                  "San Leandro",
                  ]

    # Create a function that returns the position of the first matching substring, or a large number if none match
    def get_sort_key(dataset_name):
        for i, substring in enumerate(city_order):
            if substring in dataset_name:
                return i
        return len(city_order)  # If no match, put at the end

    mean_distances['sort_key'] = mean_distances['Dataset'].apply(get_sort_key)
    mean_distances = mean_distances.sort_values('sort_key').drop('sort_key', axis=1)

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
        'Borda',
        'F1',
        'Plurality',
        'Plurality + Veto',
        'Two Approval',
        'IRV'
    ]

    # Get unique rule_names for coloring
    unique_rules = mean_distances['rule_name'].unique()
    unique_datasets = mean_distances['Dataset'].unique()


    legend_beside_plot = False
    if legend_beside_plot:
        fig = plt.figure(figsize=(14, 4.5))
    else:
        fig = plt.figure(figsize=(10, 4.5))

    x_positions = np.arange(len(unique_datasets))
    dataset_to_x = dict(zip(unique_datasets, x_positions))

    # x_label_dict = {x: x_axis_labels[name] for name, x in dataset_to_x.items()}
    x_label_dict = {x: name for name, x in dataset_to_x.items()}
    x_labels = [x_label_dict[x] for x in x_label_dict.keys()]

    # map election index to tuples of (max_dist, n_voters, n_candidates, min_dist)
    election_size_info = {}

    # Plot each rule_name with different colors and markers
    # for rule in unique_rules:
    for _, row in mean_distances.iterrows():
        x_pos = dataset_to_x[row['Dataset']]
        rule = row['rule_name']
        distance = row['distance']
        n_voters = row['n_voters']
        n_alternatives = row['n_alternatives']
        # if len(election_size_info) == 0 or election_size_info[-1] != (row['distance'], n_voters, n_alternatives):
        if x_pos not in election_size_info:
            election_size_info[x_pos] = (distance, n_voters, n_alternatives, distance)
        else:
            if election_size_info[x_pos][0] < distance:
                # update largest distance for this election
                election_size_info[x_pos] = (distance, n_voters, n_alternatives, election_size_info[x_pos][3])

            if election_size_info[x_pos][3] > distance:
                # update smallest distance for this election
                election_size_info[x_pos] = (election_size_info[x_pos][0], n_voters, n_alternatives, distance)
        # election_size_info.append((row['distance'], n_voters, n_alternatives))

        if rule in rule_renaming_map:
            rule = rule_renaming_map[rule]
        if rule not in rules_to_display:
            continue

        # colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        colour = plt_util.get_consistent_color(rule,
                                               cache=rule_colour_dict)
        marker = rule_marker_dict[rule] if rule in rule_marker_dict else plt_util.get_consistent_marker(rule)
        # mask = mean_distances['rule_name'] == rule
        plt.scatter(
            x_pos,
            row['distance'],
            color=colour,
            marker=marker,
            s=90,
            label=rule,
            alpha=1
        )

    # Add some text showing the number of voters/candidates
    if show_num_voters_and_cands:
        for x_pos in range(len(election_size_info)):
            height, n_voters, n_alternatives, min_dist = election_size_info[x_pos]
            plt.text(x_pos, height+0.005, f"{n_voters}\n{n_alternatives}", ha='center', va='bottom', size=9)

    print(f"Printing labels and indices of preflib city elections:")
    for idx, label in enumerate(x_labels):
        height, n_voters, n_alternatives, min_dist = election_size_info[idx]
        print(f"{idx}, {label}, {n_voters}, {n_alternatives}, {min_dist}")

    # plt.xticks(x_positions, x_labels, rotation=45, ha='right')

    plt.ylim((-0.005, 0.17))

    # Add labels and title
    plt.gca().tick_params(axis='both', which='major', labelsize=20)
    plt.ylabel('Distance', fontsize=26)
    plt.xlabel('Election', fontsize=26)

    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().tick_params(axis='x', which='minor', bottom=True)

    if legend_beside_plot:
        handles, labels = plt_util.organize_legend_handles(plt.gca())
        plt.legend(handles, labels, ncols=1, bbox_to_anchor=(1.335, 1.03), fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 1])
    else:
        handles, labels = plt_util.organize_legend_handles(plt.gca())
        # plt.legend(handles, labels, ncols=6, bbox_to_anchor=(0.4, 0.01), fontsize=15)
        fig.legend(handles, labels, ncols=3, loc="outside lower center", fontsize=20)
        # fig.legend(handles, labels, ncols=6,
        #            loc="upper center",
        #            bbox_to_anchor=(0.53, 0.97),
        #            fontsize=14.5)
        plt.grid(True, alpha=0.3)
        # plt.subplots_adjust(bottom=0.5)
        plt.tight_layout(rect=[0, 0.2, 1, 1])


    # Show the plot
    # plt.tight_layout(rect=[0, 0, 1, 1])
    # plt.show()
    plt.savefig("preflib/plots/preflib_scatter.png")


def bar_plot_preflib():
    # Load the CSV file
    df = pd.read_csv('preflib/analysis_results-neurips.csv')
    # df2 = pd.read_csv('preflib/olympics_data.csv')
    # # df2_renamed = df2.copy()
    # df2.columns = df.columns[:len(df2.columns)]
    # df = pd.concat([df, df2], ignore_index=True)

    # Define the list of rule_names you want to show individually
    # Replace these with your actual rule names of interest
    # specific_rules = ['Borda', 'Plurality', 'Plurality Veto', 'Anti-Plurality', "Annealing Score Vector",
    #                   'Two Approval']

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
        'Borda',
        'Plurality',
        'Plurality + Veto',
        'Two Approval',
        'IRV'
    ]

    # color_dict = {
    #     'Annealing Score Vector': [0.89019608, 0.46666667, 0.76078431, 1.],
    #     'Anti-Plurality': [0.83921569, 0.15294118, 0.15686275, 1.],
    #     'Borda': [0.12156863, 0.46666667, 0.70588235, 1.],
    #     'Empirical Rule': [0.54901961, 0.3372549, 0.29411765, 1.],
    #     'Plurality': [1., 0.49803922, 0.05490196, 1.],
    #     'Plurality Veto': [0.17254902, 0.62745098, 0.17254902, 1.],
    #     'Two Approval': [0.58039216, 0.40392157, 0.74117647, 1.]
    # }

    # Create a new column to group by
    df['rule_group'] = df['rule_name'].apply(lambda x: x if x in rules_to_display else 'IRV')

    # Calculate average distance for each rule group
    avg_by_rule = df.groupby('rule_group')['distance'].mean().reset_index()

    # Separate 'Empirical Rule' and specific rules
    other_rules = avg_by_rule[avg_by_rule['rule_group'] == 'IRV']
    specific_rules_data = avg_by_rule[avg_by_rule['rule_group'] != 'IRV']

    # Sort specific rules by increasing distance
    specific_rules_data = specific_rules_data.sort_values('distance')

    # Combine back with 'Empirical Rule' first
    if not other_rules.empty:
        avg_by_rule = pd.concat([other_rules, specific_rules_data]).reset_index(drop=True)
    else:
        avg_by_rule = specific_rules_data

    color_dict = {
        # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        rule: plt_util.get_consistent_color(rule,
                                            cache=rule_colour_dict)
        for rule in avg_by_rule['rule_group']
    }
    colors = [c for rule, c in color_dict.items()]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3, axis="y")
    bars = plt.bar(avg_by_rule['rule_group'], avg_by_rule['distance'], color=colors)

    # # Add textures/hatching to the bars
    # for bar, hatch in zip(bars, hatches[:len(bars)]):
    #     bar.set_hatch(hatch)

    # Add labels and title
    # plt.xlabel('Voting Rule', fontsize=14)
    # plt.ylabel('Split Distance', fontsize=14)
    # axes[idx].tick_params(axis='both', labelsize=16)
    # plt.title('Split Distance on Empirical Data', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    # plt.ylim((0, 0.1))
    plt.gca().tick_params(axis='both', which='major', labelsize=16)

    # Add the actual average values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02 * max(avg_by_rule['distance']),
                 f'{height:.2f}', ha='center', va='bottom', size=13)

    plt.tight_layout()
    # plt.show()
    plt.savefig("preflib/plots/preflib_bar.png")


def count_min_distance_ties(data="preflib"):
    """

    :param data:
    :return:
    """
    # Load the CSV file
    if data == "preflib":
        filename = "preflib/analysis_results-neurips.csv"
    elif data == "olympics":
        filename = "results/olympic_data-neurips.csv"
    else:
        raise ValueError("Bad passed value.")
    df = pd.read_csv(filename)

    if data == "olympics":
        df = df.rename(columns={"Game": "Dataset"})

    # Initialize counters
    min_rule_counts = Counter()
    tie_count = 0
    total_count = 0

    non_empirical_rules = {
        "Borda",
        "Plurality Veto",
        "Two Approval",
        "Veto",
        "Single Profile Annealing",
        "Plurality",
    }
    empirical_tie_counts = 0

    # Process each unique Dataset value
    for dataset, group in df.groupby('Dataset'):
        # Find minimum distance for this dataset
        min_distance = group['distance'].min()

        # Find all rows with the minimum distance for this dataset
        min_rows = group[group['distance'] == min_distance]

        # Count occurrence of each rule_name in the minimum rows
        for rule in min_rows['rule_name']:
            min_rule_counts[rule] += 1

            if data == "preflib" and rule not in non_empirical_rules:
                # Only ever one empirical rule per dataset so this shouldn't double count
                empirical_tie_counts += 1

        # Check if there's a tie (more than one row with minimum distance)
        if len(min_rows) > 1:
            tie_count += 1
        total_count += 1

    print("Frequency of each rule_name having the minimal distance:")
    for rule, count in min_rule_counts.most_common():
        print(f"{rule}: {count}")

    print(f"\nTotal number of datasets with ties for minimum distance: {tie_count}")
    print(f"Total number of datasets: {total_count}")

    print(f"Empirical rule is tied in {empirical_tie_counts} elections.")


def alma_bar_plot():
    file_name = "alma_data_cycle10"
    # file_name = "alma_output"
    folder = "alma_data"
    data_path = f"{folder}/results-{file_name}.csv"
    df = pd.read_csv(data_path)

    rule_names = df["rule_name"]
    distances = df["distance"]
    dist_std = df["distance_std"]
    data = list(zip(rule_names, distances, dist_std))

    rule_order = ["PL MLE", "Single Profile Annealing", "Borda", "Plurality Veto", "Two Approval", "Plurality", "Veto"]
    data.sort(key=lambda x: rule_order.index(x[0]))
    rule_names, distances, dist_std = [list(t) for t in zip(*data)]

    # rename before getting colour
    rule_names = [rn if rn != "Plurality Veto" else "Plurality + Veto" for rn in rule_names]
    rule_names = [rn if rn != "Single Profile Annealing" else "Best Positional Scores" for rn in rule_names]
    color_dict = {
        # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        rule: plt_util.get_consistent_color(rule,
                                            cache=rule_colour_dict)
        for rule in rule_names
    }
    colors = [c for rule, c in color_dict.items()]

    # rename after getting colour...
    rule_names = [rn if rn != "Best Positional Scores" else "Best Positional\nScores" for rn in rule_names]

    # # Update annealing rule name to fit better
    # mean_distances['rule_name'] = mean_distances['rule_name'].apply(
    #     lambda x: x if x != "Best Positional Scores" else "Best Positional\nScores")

    plt.figure(figsize=(10, 4.5))
    plt.grid(True, alpha=0.3, axis="y")
    bars = plt.bar(
        rule_names, distances,
        yerr=dist_std,
        color=colors,
        # error_kw={'elinewidth': 1.5, 'alpha': 0.5}
    )

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Distance", fontsize=14)
    plt.ylim((0.06, 0.13))
    plt.gca().tick_params(axis='both', which='major', labelsize=16)

    # Add the actual average values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.04 * max(distances),
                 f'{height:.3f}', ha='center', va='bottom', size=16)

    plt.tight_layout()

    # plt.show()
    plt.savefig(f"{folder}/{file_name}-distances.png")



if __name__ == "__main__":
    # bar_plot_f1()
    # scatter_plot_f1()
    # scatter_plot_olympics()
    # bar_plot_olympics()
    # bar_plot_preflib()
    # scatter_plot_preflib(
    #     include_zero_valued_elections=False,
    #     exclude_elections_above_max_y=0.3,
    #     show_num_voters_and_cands=False
    # )

    # count_min_distance_ties()

    alma_bar_plot()
