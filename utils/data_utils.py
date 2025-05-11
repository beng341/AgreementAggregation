import copy
import itertools
import random

import pingouin as pg
from statsmodels.multivariate.manova import MANOVA
import pandas as pd
import prefsampling as ps
import numpy as np
from scipy.stats import gamma, rankdata
from scipy import stats
import tensorflow_probability as tfp

from collections import Counter

import os
import json
import csv


def generate_profiles(distribution, profiles_per_distribution, num_voters, num_candidates, candidates_per_voter, assignments, seed=None, *args, **kwargs):
    if isinstance(distribution, str):
        distribution = [distribution]

    if not hasattr(distribution, "__iter__"):
        raise ValueError(f"distribution must be string or iterable. Given: {distribution}")

    all_profiles = []

    for dist in distribution:
        profiles = [generate_profile(distribution=dist,
                                     num_voters=num_voters,
                                     num_candidates=candidates_per_voter,
                                     seed=seed,
                                     *args,
                                     **kwargs) for _ in range(profiles_per_distribution)]

        # map profiles to assignments
        new_profiles = []
        for profile in profiles:
            prof = []
            for v in range(kwargs["n"]):
                to_rank = assignments[v].nonzero()[0]   # the "true" ranking we sample from
                order = profile[v]
                new_order = [to_rank[o] for o in order]
                prof.append(new_order)
            new_profile = np.asarray(prof)

            # alt_counts = [0 for _ in range(kwargs["m"])]
            # for alternative in range(kwargs["m"]):
            #     alt_counts[alternative] = np.sum(new_profile == alternative)

            # new_profiles.append(new_profile)
            all_profiles.append(new_profile)

        # all_profiles += profiles
    return all_profiles


def generate_profile(distribution, num_voters, num_candidates, seed=None, *args, **kwargs):
    """

    :param distribution:
    :param seed:
    :param num_voters:
    :param num_candidates:
    :param args:
    :param kwargs:
    :return:
    """
    if distribution == "identity":
        prof = ps.ordinal.identity(num_voters=num_voters,
                                   num_candidates=num_candidates,
                                   seed=seed, *args, **kwargs)
    elif distribution == "IC":
        prof = ps.ordinal.impartial(num_voters=num_voters,
                                    num_candidates=num_candidates,
                                    seed=seed, *args, **kwargs)
    elif distribution == "IAC":
        prof = ps.ordinal.impartial_anonymous(num_voters=num_voters,
                                              num_candidates=num_candidates,
                                              seed=seed, *args, **kwargs)
    elif distribution == "single_peaked_conitzer":
        prof = ps.ordinal.single_peaked_conitzer(num_voters=num_voters,
                                                 num_candidates=num_candidates,
                                                 seed=seed, *args, **kwargs)
    elif distribution == "single_peaked_walsh":
        prof = ps.ordinal.single_peaked_walsh(num_voters=num_voters,
                                              num_candidates=num_candidates,
                                              seed=seed, *args, **kwargs)
    elif distribution == "MALLOWS-RELPHI-R":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=None, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-RELPHI-0.99":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.99, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)

    elif distribution == "MALLOWS-RELPHI-0.9":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.9, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-RELPHI-0.5":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.5, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-RELPHI-0.4":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.4, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-0.9":
        impartial_central_vote = False
        central_vote = [i for i in range(num_candidates)]

        # phi = phi_from_relphi(num_candidates, relphi=0.4, seed=seed)
        normalise_phi = False
        phi = 0.9

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-0.4":
        impartial_central_vote = False
        central_vote = [i for i in range(num_candidates)]

        # phi = phi_from_relphi(num_candidates, relphi=0.4, seed=seed)
        normalise_phi = False
        phi = 0.4

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
        # prof = torch.tensor(mk.sample(m=num_candidates, n=num_voters, phi=phi))
    elif distribution == "MALLOWS-0.1":
        impartial_central_vote = False
        central_vote = [i for i in range(num_candidates)]

        # phi = phi_from_relphi(num_candidates, relphi=0.4, seed=seed)
        normalise_phi = False
        phi = 0.1

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-RELPHI-0.1":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.1, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "MALLOWS-RELPHI-0.01":
        impartial_central_vote = False
        if 'normalise_phi' in kwargs:
            normalise_phi = kwargs['normalise_phi']
        else:
            normalise_phi = False

        central_vote = [i for i in range(num_candidates)]

        phi = phi_from_relphi(num_candidates, relphi=0.01, seed=seed)

        prof = ps.ordinal.mallows(num_voters,
                                  num_candidates,
                                  phi,
                                  normalise_phi=normalise_phi,
                                  central_vote=central_vote,
                                  impartial_central_vote=impartial_central_vote,
                                  seed=seed)
    elif distribution == "URN-R":

        rng = np.random.default_rng(seed)
        # alpha = rounding(math.factorial(num_candidates) * gamma.rvs(0.8, random_state=rng))
        alpha = gamma.rvs(0.8, random_state=rng)
        prof = ps.ordinal.urn(num_voters,
                              num_candidates,
                              alpha,
                              seed=seed)

    elif distribution == "plackett_luce":

        if 'alphas' not in kwargs:
            # if pref_dist == "plackett_luce":
            alpha = 0.5
            alphas = [np.exp(alpha * i) for i in range(num_candidates, 0, -1)]
            # args["alphas"] = alphas
            # raise ValueError(
            #     "Error: alphas parameter missing.  A value must be specified for each candidate indicating their relative quality.")
        else:
            alphas = kwargs['alphas']

        prof = ps.ordinal.plackett_luce(num_voters,
                                        num_candidates,
                                        alphas,
                                        seed=seed)
    elif distribution == "plackett_luce-R":

        if 'alphas' not in kwargs:
            # Generate alpha values semi-randomly
            alpha = 0.5
            products = [num_candidates]
            for nc in range(1, num_candidates):
                products.append(random.uniform(products[-1], 0))
            alphas = [np.exp(alpha * products[i]) for i in range(num_candidates)]
            # alphas = [np.exp(alpha * i) for i in range(num_candidates, 0, -1)]
            # args["alphas"] = alphas
            # raise ValueError(
            #     "Error: alphas parameter missing.  A value must be specified for each candidate indicating their relative quality.")
        else:
            alphas = kwargs['alphas']

        prof = ps.ordinal.plackett_luce(num_voters,
                                        num_candidates,
                                        alphas,
                                        seed=seed)
    elif distribution == "plackett_luce-tf":

        # generate list where each voter's ranking is replaced with the score of that rank
        alpha = 0.5
        alphas = [np.exp(alpha * i) for i in range(num_candidates, 0, -1)]

        all_rankings = []

        for v in range(num_voters):
            scores = [alphas[c] for c in range(num_candidates)]
            ranking = tfp.distributions.PlackettLuce(scores)
            v_ranking = ranking.sample()
            # r_review = [to_review[r][i] for i in r_ranking]
            # reviews[r] = r_review
            all_rankings.append(v_ranking.numpy().tolist())
        prof = all_rankings

        # scores = []
        # to_review = [[p for p in range(m) if assignments[p, r] == 1] for r in range(n)]     # 3, 5, 7
        # reviews = [[0] * k for _ in range(n)]
        # for r in range(n):
        #     scores_to_review = [scores[i] for i in to_review[r]]        # scores[3], score[5], score[7]
        #     ranking = tfp.distributions.PlackettLuce(scores_to_review)
        #     r_ranking = ranking.sample()
        #     r_review = [to_review[r][i] for i in r_ranking]
        #     reviews[r] = r_review
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return np.array(prof)


def _mallows_profile(num_voters, num_candidates, phi, seed, normalize=True, **kwargs):
    """
    Generate mallows profile, always using the same reference ranking of 0 through m-1.
    :param num_voters:
    :param num_candidates:
    :param phi:
    :param normalize:
    :return:
    """
    impartial_central_vote = False
    if 'normalise_phi' in kwargs:
        normalise_phi = kwargs['normalise_phi']
    else:
        normalise_phi = False

    reference_ranking = [i for i in range(num_candidates)]

    if normalize:
        phi = phi_from_relphi(num_candidates, relphi=None, seed=seed)

    prof = ps.ordinal.mallows(num_voters,
                              num_candidates,
                              phi,
                              normalise_phi=normalise_phi,
                              central_vote=reference_ranking,
                              impartial_central_vote=False,  # overrides the reference ranking if True
                              seed=seed)
    return prof


def count_rankings_in_splits(s1, s2, m):
    """
    Count the number of times in each profile split that alternatives i is ranked in position j.
    Also count for the entire set of profiles.
    :param s1:
    :param s2:
    :param m:
    :return:
    """

    # (n, m, k, l)
    # n - number of voters (?)
    # m - number of alternatives (?)
    # k - ???
    # l - number of alternatives rankings per voter ("# number of proposals per reviewer")

    M1 = np.zeros((m, m))  # M1_jp = how many reviewers in set1 ranked proposal p on position j
    M2 = np.zeros((m, m))  # M2_jp = how many reviewers in set2 ranked proposal p on position j
    M_sum = np.zeros((m, m))  # total rankings of proposal p on position j
    # M_sum = torch.Tensor([[0] * m for _ in range(k)])  # in total
    M = [M1, M2, M_sum]
    proposal_split = np.zeros((m, 2))

    rankings = [s1, s2]
    for i in [0, 1]:
        for order in rankings[i]:
            for alt_rank, alternative in enumerate(order):
                M[i][alternative][alt_rank] += 1  # current split
                M[2][alternative][alt_rank] += 1  # total count of rankings
                proposal_split[alternative][i] += 1  # number of times alternatives appears in current split

    return M, proposal_split


# TAKEN FROM PREF_VOTING LIBRARY
# Given the number m of candidates and a phi in [0,1],
# compute the expected number of swaps in a vote sampled
# from the Mallows model
def find_expected_number_of_swaps(num_candidates, phi):
    res = phi * num_candidates / (1 - phi)
    for j in range(1, num_candidates + 1):
        res = res + (j * (phi ** j)) / ((phi ** j) - 1)
    return res


# TAKEN FROM PREF_VOTING LIBRARY
# Given the number m of candidates and a absolute number of
# expected swaps exp_abs, this function returns a value of
# phi such that in a vote sampled from Mallows model with
# this parameter the expected number of swaps is exp_abs
def phi_from_relphi(num_candidates, relphi=None, seed=None):
    rng = np.random.default_rng(seed)
    if relphi is None:
        relphi = rng.uniform(0.001, 0.999)
    if relphi == 1:
        return 1
    exp_abs = relphi * (num_candidates * (num_candidates - 1)) / 4
    low = 0
    high = 1
    while low <= high:
        mid = (high + low) / 2
        cur = find_expected_number_of_swaps(num_candidates, mid)
        if abs(cur - exp_abs) < 1e-5:
            return mid
        # If x is greater, ignore left half
        if cur < exp_abs:
            low = mid

        # If x is smaller, ignore right half
        elif cur > exp_abs:
            high = mid

    # If we reach here, then the element was not present
    return -1


def improve_alternatives(alternatives, profile, probability):
    """
    Increase the ranking of the specified alternatives in each preference order within the profile, probabilistically.
    Increase the rank of each alternative by 1 with given probability, if it is possible to do so.
    (Strictly speaking, this decreases rank -- it moves the specified alternatives closer to being a voter's favourite.)
    :param alternatives: tuple containing one or more tied alternatives.
    :param profile:
    :param probability:
    :return:
    """

    for order in profile:
        if random.uniform(0, 1) < probability:
            for alt in alternatives:
                idx = np.argwhere(order == alt)[0]
                # idx = order.index(alternatives)
                if idx > 0:
                    order[idx - 1], order[idx] = order[idx], order[idx - 1]
    return profile


def multiply_profile(profile, k):
    """
    Replace each preference order in a profile with k copies of the same order and return the resulting profile.
    :param profile:
    :param k:
    :return:
    """

    expanded_profile = []
    for order in profile:
        for _ in range(k):
            expanded_profile.append(copy.copy(order))

    return np.array(expanded_profile)


def normalize_score_vector(vec):
    """
    Normalize so that the highest value is 1 and the lowest value is 0.
    :param vec: list containing score "vector"
    :return:
    """
    if min(vec) == max(vec):
        return [0 for _ in vec]
    to_subtract = min(vec)  # subtract this from all values to get the lowest score to zero and all values positive
    vec = [v - to_subtract for v in vec]

    # get max value to 1 and others suitably scaled
    vec = [round(v / max(vec), 3) for v in vec]

    return vec


def sort_by_indices(values, indices):   # Claude AI
    """
    Sort values based on target indices, grouping ties into tuples.

    Args:
        values: List of values to be sorted
        indices: List of target indices for each value

    Returns:
        List with values sorted according to indices, ties grouped in tuples
    """
    # Create pairs of (index, value)
    pairs = list(zip(indices, values))

    # Group by index
    index_groups = {}
    for idx, val in pairs:
        if idx not in index_groups:
            index_groups[idx] = []
        index_groups[idx].append(val)

    # Create result list with single values or tuples
    result = [None] * len(index_groups)
    for idx, vals in index_groups.items():
        if len(vals) == 1:
            result[idx] = (vals[0], )
        else:
            result[idx] = tuple(vals)

    return result


def rename_candidates(voter_rankings):
    """

    :param voter_rankings: List of lists each containing a pair of (candidate, ranking). Rename so each candidate
    is an int starting from 0 and going up sequentially.
    :return:
    """
    all_candidates = itertools.chain.from_iterable(voter_rankings)
    all_candidates = set([x[0] for x in all_candidates])

    cand_map = {c: idx for idx, c in enumerate(list(all_candidates))}

    voter_rankings = [[(cand_map[c], r) for (c, r) in ranking] for ranking in voter_rankings]

    return voter_rankings


def rankings_from_candidate_pair_lists(voter_rankings, strict=False):
    """
    Partially ClaudeAI generated. Convert each individual voter ranking with candidate ids into a single ranking without
    the actual ranks.
    :param voter_rankings:
    :param strict: Return weak rankings if False, otherwise return strict rankings with ties broken arbitrarily.
    :return:
    """
    rankings = []

    if not strict:  # weak rankings
        for ranking in voter_rankings:
            # Find the maximum rank to determine the size of our result
            max_rank = max(rank for _, rank in ranking)

            # Initialize empty lists for each rank
            # We use index 0 for rank 1, index 1 for rank 2, etc.
            rank_groups = [[] for _ in range(max_rank)]

            # Group ids by their ranks
            for id_num, rank in ranking:
                rank_groups[rank - 1].append(id_num)

            # Convert lists to tuples
            rankings.append([tuple(group) for group in rank_groups])
    else:
        for ranking in voter_rankings:
            # ensure that ranking is sorted (should be already...)

            ranking.sort(key=lambda x: x[1])
            ranking = [r[0] for r in ranking]
            rankings.append(ranking)

    return rankings


def weak_ranking_from_strict_ranking(profile, m):
    """
    Given a profile of strict orders convert them to a profile with ties that puts any missing alternatives at the
    bottom of the ranking.
    Generated by Claude AI.
    :param profile:
    :param m: Total number of alternatives
    :return:
    """
    new_profile = []

    for order in profile:
        new_list = []
        # Convert each integer to a tuple
        for num in order:
            new_list.append((num,))

        # Find missing values between 0 and m
        all_values = set(range(m))
        present_values = set(order)
        missing_values = all_values - present_values

        # Add missing values as a single tuple at the end (if any exist)
        if missing_values:
            new_list.append(tuple(sorted(missing_values)))

        new_profile.append(new_list)

    return new_profile


def parse_astronomy_csv(min_num_scores=4):
    """
    Loads a CSV file and creates a dictionary mapping each unique proposal_analysis_id
    to a list of triples (reviewer_analysis_id, proposal_rank, rating).

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary with proposal_analysis_id as keys and lists of triples as values.
    """
    file_path = "reviewer_data/astronomy/reviews_coupled_with_feedback.csv"

    try:
        df = pd.read_csv(file_path)
        df = df[["proposal_analysis_id", "user_analysis_id", "grade"]]
        df = df.dropna(how="any")

        # add rank for scores provided by each user
        df['grade_rank'] = df.groupby('user_analysis_id')['grade'].rank(
            method='dense',
            ascending=True
        ).astype(int)

        proposal_scores = dict()
        proposal_ranks = dict()

        # also track the partial ranking generated by each voter
        voter_rankings = [sorted(list(zip(df[df["user_analysis_id"] == u]["proposal_analysis_id"].to_list(), df[df["user_analysis_id"] == u]["grade_rank"].to_list())), key=lambda x: x[1]) for u in df["user_analysis_id"].unique()]

        prop_ids = {p: i for i, p in enumerate(df["proposal_analysis_id"].unique()) if len(df[df["proposal_analysis_id"] == p]) >= min_num_scores}
        # rename proposals to be continuous indices starting from zero
        for proposal in df["proposal_analysis_id"].unique():
            filtered_df = df[df["proposal_analysis_id"] == proposal]
            if len(filtered_df) >= min_num_scores:
                proposal_scores[prop_ids[proposal]] = filtered_df["grade"].to_list()
                proposal_ranks[prop_ids[proposal]] = filtered_df["grade_rank"].to_list()


    except Exception as e:
        print(f"Exception while loading review scores: {e}")

    return df, proposal_scores, proposal_ranks, voter_rankings


def load_json_folder():
    """
    Creates a list of dictionaries from JSON files in the given folder.

    Args:
        folder_path (str): The path to the folder containing JSON files.

    Returns:
        list: A list where each element is a dictionary parsed from a JSON file.
    """
    folder_path = "reviewer_data/json/ACL2017"
    json_list = []

    try:
        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            # Construct the full path to the file
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a JSON file
            if file_name.endswith('.json') and os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    try:
                        # Parse the JSON file and append it to the list
                        data = json.load(json_file)
                        json_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return json_list


def acl_2017_stats(json_list):

    stats = []
    for js in json_list:
        scores = []
        for review in js["reviews"]:
            scores.append(eval(review["RECOMMENDATION"]))

        stats.append((len(scores), np.mean(scores), min(scores), max(scores)))

    stats.sort(key=lambda x: x[0])
    review_counts = Counter()
    print("# Reviews, Mean, Min, Max")
    for stat in stats:
        print(stat)

        # track number of papers with 8 reviews, 7 reviews, etc.
        review_counts[stat[0]] += 1

    print()
    print("Number of proposals with each number of reviews")
    print(review_counts)


def astronomy_stats(astro_df):

    # proposal_scores = dict()

    review_stats = []
    for proposal in astro_df["proposal_analysis_id"].unique():
        filtered_df = astro_df[astro_df["proposal_analysis_id"] == proposal]
        # proposal_scores[proposal] = filtered_df[["grade"]].numpy()

        mean = filtered_df[["grade"]].mean()
        min_val = filtered_df[["grade"]].min()
        max_val = filtered_df[["grade"]].max()
        count = len(filtered_df)

        review_stats.append((proposal, mean, min_val, max_val, count))

    # rating_stats = []
    # ranking_stats = []
    # for proposal, reviews in astro_dict.items():
    #     ratings = []
    #     rankings = []
    #     for review in reviews:
    #         rankings.append(review[1])
    #         ratings.append(review[2])
    #
    #     ranking_stats.append((len(rankings), np.mean(rankings), max(rankings), min(rankings)))
    #     rating_stats.append((len(ratings), np.mean(ratings), max(ratings), min(ratings)))

    # ranking_stats.sort(key=lambda x: x[0])
    # rating_stats.sort(key=lambda x: x[0])
    print("# Reviews, Mean, Min, Max")
    review_counts = Counter()
    for idx in range(len(review_stats)):

        print(f"Review stats: {review_stats[idx]}")

        # track number of papers with 8 reviews, 7 reviews, etc.
        review_counts[review_stats[idx][4]] += 1

    print()
    print("Number of proposals with each number of reviews:")
    print(review_counts)


def print_pvalues():
    """
    Quick utility to print pvalues of saved test data.
    Independent variable: "voting rule"
    Dependent variables: "KT Distance Between Splits", "Distance from Central Vote"
    :return:
    """
    df = pd.read_csv("results/experiment-ground_truth_vs_split_distance-testing-nsplits=10.csv")
    filtered_df = df[df["preference distribution"] == "MALLOWS-0.4"]

    # col = "KT Distance Between Splits"
    # data = pg.anova(data=filtered_df, dv=col, between="voting rule", detailed=True)
    # print(f"Results for: {col}")
    # print(data)
    # print()
    #
    # col = "Distance from Central Vote"
    # data = pg.anova(data=filtered_df, dv=col, between="voting rule", detailed=True)
    # print(f"Results for: {col}")
    # print(data)
    # print()

    # Need to get rid of spaces in all the column names we use
    filtered_df = filtered_df.rename(columns={"KT Distance Between Splits": "Split_Distance",
                                              "Distance from Central Vote": "Reference_Distance",
                                              "voting rule": "voting_rule"})

    dvs = ["Split_Distance", "Reference_Distance"]
    between = "voting_rule"
    formula = f'{" + ".join(dvs)} ~ {between}'
    manova = MANOVA.from_formula(formula, data=filtered_df)
    manova_results = manova.mv_test()
    print(f"Results for MANOVA test (MALLOWS-0.4)")
    print(manova_results)

    print("")

    filtered_df = df[df["preference distribution"] == "plackett_luce"]
    filtered_df = filtered_df.rename(columns={"KT Distance Between Splits": "Split_Distance",
                                              "Distance from Central Vote": "Reference_Distance",
                                              "voting rule": "voting_rule"})
    dvs = ["Split_Distance", "Reference_Distance"]
    between = "voting_rule"
    formula = f'{" + ".join(dvs)} ~ {between}'
    manova = MANOVA.from_formula(formula, data=filtered_df)
    manova_results = manova.mv_test()
    print(f"Results for MANOVA test (Plackett-Luce)")
    print(manova_results)



def compare_best_distances():
    """
    For some existing data, count how often the minimum distance between splits is also the minimum distance from the
    ground truth.
    :return:
    """
    filename = "results/experiment-ground_truth_vs_split_distance-testing-nsplits=10-complete.csv"
    # filename = "results/experiment-ground_truth_vs_split_distance-testing-nsplits=3-testing.csv"
    whole_df = pd.read_csv(filename)

    pref_dists = ["MALLOWS-0.4", "plackett_luce"]

    for pref_dist in pref_dists:
        if pref_dist == "MALLOWS-0.4":
            target_voting_rule = "Kemeny"
        elif pref_dist == "plackett_luce":
            target_voting_rule = "PL MLE"
        else:
            raise ValueError("Should be impossible. What did you mess up?")

        df = whole_df[whole_df["preference distribution"] == pref_dist]

        # Get unique profile sets
        profile_sets = df['profile_set_idx'].unique()
        same_index_count = 0
        target_rule_count = 0

        for profile_idx in profile_sets:
            profile_df = df[df['profile_set_idx'] == profile_idx]

            # Find indices of minimum values
            min_kt_idx = profile_df['KT Distance Between Splits'].idxmin()
            min_central_idx = profile_df['Distance from Central Vote'].idxmin()

            # Check if both min distances are from same rule
            if min_kt_idx == min_central_idx:
                same_index_count += 1

                # Check if that rule is the MLE
                if profile_df.loc[min_kt_idx, 'voting rule'] == target_voting_rule:
                    target_rule_count += 1

        total_profiles = len(profile_sets)

        print(f"{pref_dist} preferences:")
        print(f"Both min distances have same rule: {same_index_count}/{total_profiles} = {same_index_count / total_profiles}")
        print(f"MLE ({target_voting_rule}) achieves both min distances: {target_rule_count}/{total_profiles} = {target_rule_count / total_profiles}")

        # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        corr, pval = stats.pearsonr(
            df['KT Distance Between Splits'],
            df['Distance from Central Vote']
        )

        print(f"Pearson correlation between splits and ground truth is {corr} with p={pval}")
        print()


def load_olympic_rankings(include_noncompeting_countries=True, include_nonmedaling_competitors=True, maxsize=None):
    # From Claude AI

    df = pd.read_csv('reviewer_data/olympics/athlete_events.csv')

    # Step 2: Filter to include only the required columns
    df_filtered = df[['Name', 'NOC', 'Games', 'Year', 'Event', 'Medal']]


    # # TEMPORARY FOR DEBUGGING/TESTING
    # df_filtered = df_filtered[df_filtered["Year"] == 1992]

    # Step 3: Map NOCs to integers for each Games
    def add_noc_mapping(dataframe):
        # Create a copy of the dataframe to avoid modifying the original
        df_with_mapping = dataframe.copy()

        # Add a new column for the NOC integer mapping
        df_with_mapping['NOC_ID'] = -1  # Initialize with placeholder

        # Get unique Games values
        unique_games = df_with_mapping['Games'].unique()

        # For each Games, map NOCs to integers starting from 0
        for game in unique_games:
            # Get data for this specific Games
            game_mask = df_with_mapping['Games'] == game
            game_data = df_with_mapping[game_mask]

            # Get unique NOCs for this Games
            unique_nocs = game_data['NOC'].unique()

            # Create mapping dictionary for this Games
            noc_to_id = {noc: i for i, noc in enumerate(unique_nocs)}

            # Apply mapping to the dataframe
            for noc, noc_id in noc_to_id.items():
                # Update NOC_ID for rows matching both the Games and NOC
                df_with_mapping.loc[(game_mask) & (df_with_mapping['NOC'] == noc), 'NOC_ID'] = noc_id

        return df_with_mapping

    # Apply the mapping function to create the updated dataframe
    df_with_noc_mapping = add_noc_mapping(df_filtered)


    # Get weak ranking for a specific game/sport in filtered list
    def get_medal_lists_by_event(games_data, sport_value, all_games_nocs):
        # Filter data for the specific Event value
        event_data = games_data[games_data['Event'] == sport_value]

        # Create tuples of athletes by medal type
        gold_medalists = tuple(event_data[event_data['Medal'] == 'Gold']['NOC_ID'].unique())
        silver_medalists = tuple(event_data[event_data['Medal'] == 'Silver']['NOC_ID'].unique())
        bronze_medalists = tuple(event_data[event_data['Medal'] == 'Bronze']['NOC_ID'].unique())

        # Get athletes with no medals (NaN in Medal column)
        non_medalists = tuple(event_data[event_data['Medal'].isna()]['NOC_ID'].unique())
        # non_medalists = tuple(sport_data[sport_data['Medal'] == 'NA']['NOC'].unique())

        # Get NOCs that were at the Games but didn't compete in this event
        event_nocs = set(event_data['NOC_ID'].unique())
        non_competing_nocs = tuple(all_games_nocs - event_nocs)

        if include_noncompeting_countries and include_nonmedaling_competitors:
            ranking = [gold_medalists, silver_medalists, bronze_medalists, non_medalists, non_competing_nocs]
        elif include_nonmedaling_competitors:
            ranking = [gold_medalists, silver_medalists, bronze_medalists, non_medalists]
        else:
            ranking = [gold_medalists, silver_medalists, bronze_medalists]
        # Return the list of tuples
        return ranking

    # Step 4: Create the nested dictionary with Games and Sports
    game_event_medals = {}

    # Get unique Games values
    unique_games = df_with_noc_mapping['Games'].unique()

    for game in unique_games:
        # Filter data for the current Games
        games_data = df_with_noc_mapping[df_with_noc_mapping['Games'] == game]

        # Get all NOCs present at these Games
        all_games_nocs = set(games_data['NOC_ID'].unique())

        # Get unique Sport values for this Games
        unique_events = games_data['Event'].unique()

        # Create dictionary for this Games with Event as keys
        events_dict = {}
        for event in unique_events:
            events_dict[event] = get_medal_lists_by_event(games_data, event, all_games_nocs)

        # Add to the main dictionary
        game_event_medals[game] = events_dict

        if maxsize is not None and maxsize > len(game_event_medals.keys()):
            # to make loading data for debugging faster
            print(f"Loading only {maxsize} olympic games.")
            break


    # Flatten to simply map the year to the weak rankings, losing event information
    game_medals = {
        game: [ranking for name, ranking in event.items()] for game, event in game_event_medals.items()
    }

    return game_event_medals, game_medals


if __name__ == "__main__":

    # compare_best_distances()

    olympics_event_detail, olympics_weak_rankings_dict = load_olympic_rankings()

    # print_pvalues()

    # astro_dict, proposal_scores = parse_astronomy_csv()
    # astronomy_stats(astro_dict)

    # json_folder = load_json_folder()
    # acl_2017_stats(json_folder)