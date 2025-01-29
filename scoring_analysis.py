import itertools
import pprint
import random

from utils import data_utils as du
from utils import voting_utils as vu
import rule_comparison as rc
import numpy as np
from scipy.stats import hmean, gmean, pmean, mode


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


def score_data_kt_distance(review_scores):
    """
    Given a dict containing consecutive numeric keys mapped to a list of scores, compute the mean kt distance for many different rules
    across different splits.
    NOTE: Assume that keys start from 0 and increase by 1. All keys must be present.
    :param review_scores:
    :return:
    """

    def single_split_distance(r_scores):
        data_splits = dict()

        # split reviews on each paper
        for paper, scores in r_scores.items():
            if len(scores) < 6:
                continue

            # make 1 random split and save each set of scores for later
            random.shuffle(scores)
            first_split_size = len(scores) // 2
            first_split_size = 3
            first_split = scores[:first_split_size]
            second_split = scores[first_split_size:first_split_size*2]
            # second_split = scores[first_split_size:]
            data_splits[paper] = (first_split, second_split)

            # # Only use balanced splits of reviewers
            # assert len(first_split) == len(second_split)

        split_1 = [data_splits[paper][0] for paper in range(len(data_splits)) if paper in data_splits]
        split_2 = [data_splits[paper][1] for paper in range(len(data_splits)) if paper in data_splits]

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
        "hmean": hmean,
        "gmean": gmean,
        "midrange": lambda x: (max(x) + min(x))/2,
        "product": np.prod
        # "mode": lambda x: mode(x)[0],
        # "p = -100": lambda x: pmean(x, -100),
        # # "pmean-1": lambda x: pmean(x, -1),
        # # "pmean0": lambda x: pmean(x, 0),
        # # "pmean1": lambda x: pmean(x, 1),
        # "pmean2": lambda x: pmean(x, 2),
        # "pmean3": lambda x: pmean(x, 3),
        # "pmean4": lambda x: pmean(x, 4),
        # "p = 100": lambda x: pmean(x, 100),
        # "lehmer p = -100": lambda x: lehmer_mean(x, p=-100),
        # "lehmer p = -10": lambda x: lehmer_mean(x, p=-10),
        # "lehmer p = -2": lambda x: lehmer_mean(x, p=-1),
        # "lehmer p = -1": lambda x: lehmer_mean(x, p=-1),
        # "lehmer p = 0": lambda x: lehmer_mean(x, p=0),
        # "lehmer p = 1": lambda x: lehmer_mean(x, p=1),
        # "lehmer p = 2": lambda x: lehmer_mean(x, p=2),
        # "lehmer p = 10": lambda x: lehmer_mean(x, p=10),
        # "lehmer p = 100": lambda x: lehmer_mean(x, p=100),
    }

    all_distances = {k: [] for k, v in aggregation_functions.items()}
    n_splits = 100
    for _ in range(n_splits):
        distances = single_split_distance(r_scores=review_scores)
        for k, dist in distances.items():
            all_distances[k].append(dist)

    mean_distances = {k: np.mean(v) for k, v in all_distances.items()}
    std_distances = {k: np.std(v) for k, v in all_distances.items()}

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


if __name__ == "__main__":
    astro_scores, proposal_scores, proposal_ranks = du.parse_astronomy_csv()

    print("Distances on astronomy dataset for score data:")
    score_data_kt_distance(proposal_scores)

    print("\n\n")

    print("Distances on astronomy dataset for ordinal data")
    score_data_kt_distance(proposal_ranks)

