import copy
import random

import pandas as pd
import prefsampling as ps
import numpy as np
from scipy.stats import gamma, rankdata
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
        # alpha = round(math.factorial(num_candidates) * gamma.rvs(0.8, random_state=rng))
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
        )

        proposal_scores = dict()
        proposal_ranks = dict()

        prop_ids = {p: i for i, p in enumerate(df["proposal_analysis_id"].unique()) if len(df[df["proposal_analysis_id"] == p]) >= min_num_scores}
        # rename proposals to be continuous indices starting from zero
        for proposal in df["proposal_analysis_id"].unique():
            filtered_df = df[df["proposal_analysis_id"] == proposal]
            if len(filtered_df) >= min_num_scores:
                proposal_scores[prop_ids[proposal]] = filtered_df["grade"].to_list()
                proposal_ranks[prop_ids[proposal]] = filtered_df["grade_rank"].to_list()

        # pref_orders = []

        # user_rankings = dict()
        # user_scores = dict()
        #
        # # also calculate the rank each reviewer ranked each paper
        # for reviewer in df["user_analysis_id"].unique():
        #     filtered_df = df[df["user_analysis_id"] == reviewer]
        #     # filtered_df = filtered_df[filtered_df['proposal_analysis_id'].isin(prop_ids.keys())]
        #     # TODO: filter to include only papers that have received enough reviews (?)
        #
        #     user_scores[reviewer] = filtered_df["grade"].to_list()
        #     user_rankings[reviewer] = filtered_df["grade_rank"].to_list()
            # ranks = rankdata(filtered_df["grade"])
            # scores = filtered_df["grade"].to_list()
            #
            # user_scores = filtered_df["grade"].to_list()
            # user_scores = rankdata(user_scores, method="dense")
            # user_scores -= 1    # adjust to get 0-based indexing
            #
            # reviewed_items = filtered_df["proposal_analysis_id"].to_list()
            # reviewed_items = [prop_ids[p] for p in reviewed_items]
            #
            # preference_order = tuple(sort_by_indices(values=reviewed_items, indices=user_scores))
            # # preference_order = [prop_ids[p] for p in preference_order]
            # pref_orders.append(preference_order)
            #
            # user_rankings[reviewer] = filtered_df["grade"].to_list()
            #
            # for idx, proposal in enumerate(filtered_df["proposal_analysis_id"]):
            #     if proposal not in proposal_ranks:
            #         proposal_ranks[proposal] = []
            #     proposal_ranks[proposal].append(ranks[idx])


    except Exception as e:
        print(f"Exception while loading review scores: {e}")

    return df, proposal_scores, proposal_ranks


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


if __name__ == "__main__":
    astro_dict, proposal_scores = parse_astronomy_csv()
    astronomy_stats(astro_dict)

    # json_folder = load_json_folder()
    # acl_2017_stats(json_folder)