import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from collections import Counter
import plotting as plt_util
from plotting import rule_renaming_map, rule_colour_dict, rule_marker_dict


def scatter_plot_olympics():
    # Load the CSV file
    df = pd.read_csv('results/olympic_data-neurips-updated.csv')

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
        colour = plt_util.get_consistent_color(series_name=rule,
                                               # colormap="gist_ncar",
                                               cache=plt_util.rule_colour_dict)
        colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
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
    df = pd.read_csv('results/olympic_data-neurips-updated.csv')

    # df = df.replace('Plurality Veto', 'Plurality + Veto')
    # df = df.replace('Single Profile Annealing', 'Optimized Scores')
    df = df[df["rule_name"] != "Three Approval"]

    for old_name, new_name in rule_renaming_map.items():
        df = df.replace(to_replace=old_name, value=new_name)

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
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

    color_dict = {
        rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        for rule in mean_distances['rule_name']
    }
    colors = [c for rule, c in color_dict.items()]

    # # Separate 'Empirical Rule' and specific rules
    # f1_rules = mean_distances[mean_distances['rule_name'] == 'F1']
    # other_rules = mean_distances[mean_distances['rule_name'] != 'F1']

    # Sort specific rules by increasing distance
    mean_distances = mean_distances.sort_values('mean')

    # # Combine back with 'Empirical Rule' first
    # if not other_rules.empty:
    #     mean_distances = pd.concat([f1_rules, specific_rules_data]).reset_index(drop=True)
    # else:
    #     mean_distances = specific_rules_data

    plt.figure(figsize=(10, 6))
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
    plt.ylim((0.21, 0.4))
    plt.gca().tick_params(axis='both', which='major', labelsize=11.5)

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
        colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
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

    # mean_distances = df.groupby('rule_name')['distance'].mean().reset_index()
    mean_distances = df.groupby('rule_name')['distance'].agg(['mean', 'sem']).reset_index()
    mean_distances['sem'] = mean_distances['sem'].fillna(0)

    # Separate optimized rule, F1 rules and specific rules
    optimized_rule = mean_distances[mean_distances['rule_name'].str.contains('Best')]
    f1_rules = mean_distances[mean_distances['rule_name'].str.contains('F1')]
    # other_rules = mean_distances[~mean_distances['rule_name'].str.contains('F1')]
    other_rules = mean_distances[
        ~(mean_distances['rule_name'].str.contains('F1') | mean_distances['rule_name'].str.contains('Best'))]

    # Sort specific rules by increasing distance
    f1_rules = f1_rules.sort_values("rule_name")

    rule_order = ["F1 ('91-'02)", "F1 ('03-'09)", "F1 ('10-'18)"]
    f1_rules['rule_name'] = pd.Categorical(f1_rules['rule_name'], rule_order)
    f1_rules = f1_rules.sort_values('rule_name')
    specific_rules_data = other_rules.sort_values('rule_name')
    # specific_rules_data = other_rules.sort_

    # Combine back with F1 rules first
    if not other_rules.empty:
        mean_distances = pd.concat([optimized_rule, f1_rules, specific_rules_data]).reset_index(drop=True)
    else:
        mean_distances = specific_rules_data

    color_dict = {
        # rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        rule: plt_util.get_consistent_color(rule,
                                            cache=rule_colour_dict)
        for rule in mean_distances['rule_name']
    }
    colors = [c for rule, c in color_dict.items()]

    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3, axis="y")
    bars = plt.bar(
        mean_distances['rule_name'], mean_distances['mean'],
        yerr=mean_distances["sem"],
        color=colors,
        # error_kw={'elinewidth': 1.5, 'alpha': 0.5}
    )

    # plt.title('Split Distance on F1 Races', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Rule Name", fontsize=20)
    plt.ylabel("Distance", fontsize=20)
    plt.ylim((0.08, 0.54))
    plt.gca().tick_params(axis='both', which='major', labelsize=13)

    # Add the actual average values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.03 * max(mean_distances['mean']),
                 f'{height:.2f}', ha='center', va='bottom', size=13)

    plt.tight_layout()
    # plt.show()
    plt.savefig("preflib/plots/F1_bar-filtered-optimized.png")


def scatter_plot_preflib():
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
    df['rule_name'] = df['rule_name'].apply(lambda x: "Empirical" if "City" in str(x) else x)
    df['rule_name'] = df['rule_name'].apply(lambda x: "Empirical" if "UK Labour" in str(x) else x)

    mean_distances = df.groupby(['Dataset', 'rule_name', 'n_alternatives', 'n_voters'])['distance'].mean().reset_index()
    mean_distances = mean_distances.sort_values('Dataset')

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
        'Borda',
        'F1',
        'Plurality',
        'Plurality + Veto',
        'Two Approval',
        'Empirical'
    ]

    x_axis_labels = {
        "UK Labour": "UK Labour",
        "City (2012 Oakland City Council - District 5)": "Oakland 2012",
        "City (Aspen City Council 2009)": "Aspen Council 2009",
        "City (Aspen Mayor 2009)": "Aspen Mayor 2009",
        "City (2006 Burlington Mayoral Election)": "Burlington 2006",
        "City (2009 Burlington Mayoral Election)": "Burlington 2009",
        "City (2010 Berkeley City Council - District 7)": "Berkely 2010",
    }

    # Get unique rule_names for coloring
    unique_rules = mean_distances['rule_name'].unique()
    unique_datasets = mean_distances['Dataset'].unique()

    # Create a scatter plot
    plt.figure(figsize=(14, 6))

    x_positions = np.arange(len(unique_datasets))
    dataset_to_x = dict(zip(unique_datasets, x_positions))

    # x_label_dict = {x: x_axis_labels[name] for name, x in dataset_to_x.items()}
    x_label_dict = {x: name for name, x in dataset_to_x.items()}
    x_labels = [x_label_dict[x] for x in x_label_dict.keys()]

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
            election_size_info[x_pos] = (distance, n_voters, n_alternatives)
        else:
            if election_size_info[x_pos][0] < distance:
                election_size_info[x_pos] = (distance, n_voters, n_alternatives)
        # election_size_info.append((row['distance'], n_voters, n_alternatives))

        if rule in rule_renaming_map:
            rule = rule_renaming_map[rule]
        if rule not in rules_to_display:
            continue

        colour = rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
        marker = rule_marker_dict[rule] if rule in rule_marker_dict else plt_util.get_consistent_marker(rule)
        # mask = mean_distances['rule_name'] == rule
        plt.scatter(
            x_pos,
            row['distance'],
            color=colour,
            marker=marker,
            s=40,
            label=rule,
            alpha=0.7
        )

    for x_pos in range(len(election_size_info)):
        height, n_voters, n_alternatives = election_size_info[x_pos]
        plt.text(x_pos, height, f"{n_voters}\n{n_alternatives}", ha='center', va='bottom', size=9)

    # plt.xticks(x_positions, x_labels, rotation=45, ha='right')

    # plt.xlim((0.015, 0.145))
    # plt.ylim((-0.01, 0.18))

    # Add labels and title
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel('Distance', fontsize=16)
    plt.xlabel('Election', fontsize=16)
    # plt.title('Split Distance of Political Elections', fontsize=18)

    handles, labels = plt_util.organize_legend_handles(plt.gca())

    # map each unique label to a corresponding handle, doesn't matter which of the matching handles

    plt.legend(handles, labels, ncols=1, bbox_to_anchor=(1.28, 1.024), fontsize=15)
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 1])
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

    rules_to_display = [
        'Best Positional Scores',
        'Veto',
        'Borda',
        'Plurality',
        'Plurality + Veto',
        'Two Approval',
        'Empirical'
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
    df['rule_group'] = df['rule_name'].apply(lambda x: x if x in rules_to_display else 'Empirical Rule')

    # Calculate average distance for each rule group
    avg_by_rule = df.groupby('rule_group')['distance'].mean().reset_index()

    # Separate 'Empirical Rule' and specific rules
    other_rules = avg_by_rule[avg_by_rule['rule_group'] == 'Empirical Rule']
    specific_rules_data = avg_by_rule[avg_by_rule['rule_group'] != 'Empirical Rule']

    # Sort specific rules by increasing distance
    specific_rules_data = specific_rules_data.sort_values('distance')

    # Combine back with 'Empirical Rule' first
    if not other_rules.empty:
        avg_by_rule = pd.concat([other_rules, specific_rules_data]).reset_index(drop=True)
    else:
        avg_by_rule = specific_rules_data

    color_dict = {
        rule: rule_colour_dict[rule] if rule in rule_colour_dict else plt_util.get_consistent_color(rule)
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


if __name__ == "__main__":
    bar_plot_f1()
    scatter_plot_f1()
    scatter_plot_olympics()
    bar_plot_olympics()
    bar_plot_preflib()
    scatter_plot_preflib()

    # count_min_distance_ties()
