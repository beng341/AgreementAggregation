import itertools
import pprint
import random
from collections import Counter

from utils import data_utils as du
from utils import voting_utils as vu
import rule_comparison as rc
import numpy as np
from scipy.stats import hmean, gmean, pmean, mode, sem


def lehmer_mean(values, p):
    """
    Computes the Lehmer mean of a list of values.

    Parameters:
        values (list or array-like): A list of positive numeric values.
        p (float): The Lehmer mean parameter. Determines the type of mean.

    Returns:
        float: The Lehmer mean of the input values.
    """
    if not values:
        raise ValueError("The values list cannot be empty.")
    if any(v < 0 for v in values):
        raise ValueError("All values must be non-negative.")

    numerator = sum(v ** p for v in values)
    denominator = sum(v ** (p - 1) for v in values)

    if denominator == 0:
        raise ValueError("Denominator is zero, check the input values and parameter p.")

    return numerator / denominator


def tmean(vals):
    """
    Trimmed mean - Mean after removing the minimum and maximum values
    :param vals:
    :return:
    """
    if len(vals) <= 2:
        raise ValueError(f"Too few values to test. Got only {len(vals)}")
    svals = sorted(vals)
    # return np.mean(vals)
    return np.mean(svals[1:-1])


def score_data_kt_distance(review_scores):
    """
    Given a dict containing consecutive numeric keys mapped to a list of scores, compute the mean kt distance for many different rules
    across different splits.
    NOTE: Assume that keys start from 0 and increase by 1. All keys must be present.
    :param review_scores:
    :return:
    """

    for _, scores in review_scores.items():
        scores.sort()

    def single_split_distance(r_scores):
        data_splits = dict()

        # split reviews on each paper
        for paper, scores in r_scores.items():
            if len(scores) < 6:
                continue

            trim = False
            if trim:
                scores = scores[1:-1]

            # make 1 random split and save each set of scores for later
            random.shuffle(scores)
            first_split_size = len(scores) // 2
            # first_split_size = 3
            first_split = scores[:first_split_size]
            second_split = scores[first_split_size:first_split_size*2]
            # second_split = scores[first_split_size:]
            data_splits[paper] = (first_split, second_split)

            # # Only use balanced splits of reviewers
            # assert len(first_split) == len(second_split)

        # split_1 = [data_splits[paper][0] for paper_idx, paper in range(len(data_splits)) if paper in data_splits]
        # split_2 = [data_splits[paper][1] for paper in range(len(data_splits)) if paper in data_splits]
        split_1 = [data_splits[paper][0] for paper in data_splits.keys() if paper in data_splits]
        split_2 = [data_splits[paper][1] for paper in data_splits.keys() if paper in data_splits]

        # get scores and distances for each review
        distances = dict()
        for func_name, agg_func in aggregation_functions.items():
            # turn each split into weak ranking

            scores_1 = [agg_func(s1) for s1 in split_1]
            scores_2 = [agg_func(s2) for s2 in split_2]

            wr1 = vu.scores_to_tuple_ranking(scores_1)
            wr2 = vu.scores_to_tuple_ranking(scores_2)

            dist = rc.kt_distance_between_rankings(wr1, wr2)

            distances[func_name] = dist

        return distances

    aggregation_functions = {
        "mean": np.mean,
        "min": min,
        "max": max,
        "median": np.median,
        # "hmean": hmean,
        "gmean": gmean,
        "tmean": tmean
        # "midrange": lambda x: (max(x) + min(x))/2,
        # "product": np.prod
    }

    all_distances = {k: [] for k, v in aggregation_functions.items()}
    n_splits = 1000
    for _ in range(n_splits):
        distances = single_split_distance(r_scores=review_scores)
        for k, dist in distances.items():
            all_distances[k].append(dist)

    review_counts = Counter()
    for paper, scores in review_scores.items():
        review_counts[len(scores)] += 1
    # print(f"X:y; y proposals have X reviewers")
    # print(review_counts)
    # print()

    mean_distances = {k: np.mean(v) for k, v in all_distances.items()}
    std_distances = {k: np.std(v) for k, v in all_distances.items()}
    std_distances = {k: sem(v) for k, v in all_distances.items()}

    results = {k: f"{round(mean_distances[k], 3)} ± {round(std_distances[k], 3)}" for k in aggregation_functions.keys()}

    line1 = [f"{k}" for k in aggregation_functions.keys()]
    line2 = [f"${round(mean_distances[k], 3)} \pm {round(std_distances[k], 3)}$" for k in aggregation_functions.keys()]

    line1 = [" & "] + line1
    line2 = ["Astronomy & "] + line2

    line1 = " & ".join(line1) + " \\\\"
    line2 = " & ".join(line2)
    print(line1)
    print(line2)

    pprint.pprint(results)


def ordinal_data_kt_distance(voter_rankings):
    """

    :param voter_rankings: List of lists of pairs where first element is candidate id (in need of normalization) and
    second element is the rank assigned by the voter to that candidate.
    :return:
    """

    def single_rule_distance(rule, profile, num_splits):
        dist, std = rc.kt_distance_one_profile_one_rule(profile=profile, n_splits=ns, rule=rule)
        return dist, std

    ns = 200

    # normalize candidate ids
    voter_rankings = du.rename_candidates(voter_rankings)
    # convert to weak rankings without candidate ids
    rankings = du.rankings_from_candidate_pair_lists(voter_rankings, strict=True)

    # truncate to include only a random 6 rankings from each voter
    # make sure to preserve order of underlying rankings
    filtered_rankings = []
    for ranking in rankings:
        min_size = 6
        if len(ranking) < min_size:
            continue
        indices = sorted(random.sample(range(len(ranking)), min_size))
        ranking = [ranking[i] for i in indices]
        filtered_rankings.append(ranking)

    all_rules = [
        # vu.annealing_ranking,
        vu.borda_minmax_ranking,
        vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.borda_ranking,
        vu.two_approval_ranking,
        vu.kemeny_gurobi_lazy,
        # vu.choix_pl_ranking,
        vu.copeland_ranking,
        # vu.dowdall_ranking,
        # vu.three_approval,
        # vu.four_approval,
        # vu.five_approval,
        # vu.six_approval,
        # vu.random_ranking,
    ]

    all_rule_distances = dict()
    print(f"KT Distances for {ns} splits")
    for rule in all_rules:
        dist, std = rc.kt_distance_one_profile_one_rule(profile=filtered_rankings, n_splits=ns, rule=vu.borda_ranking)

        all_rule_distances[rule.name] = (dist, std)

        print(f"{round(dist, 5)} ± {round(std, 6)} is KT Distance of {rule.name}.")



if __name__ == "__main__":
    astro_scores, proposal_scores, proposal_ranks, voter_rankings = du.parse_astronomy_csv()
    score_data_kt_distance(proposal_scores)