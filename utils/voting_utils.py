import pprint
import time

import pref_voting
import torch
import numpy as np
import random
import choix

import rule_comparison
from utils.decorator import method_name
# from annealing import ScoreVectorAnnealer
from annealing import SingleProfileScoreVectorAnnealer
import sdopt_tearing_master.grb_lazy as gl
import sdopt_tearing_master.grb_pcm as gpcm
import networkx as nx

use_gurobi = True
if use_gurobi:
    import gurobipy as gp
    from gurobipy import GRB


def generate_assignments(n, m, k, l):
    """
    Generate assignment matrix of which voter votes on which candidates.
    Alternatively, think of as assignments of reviewers to papers.
    :param n: number of voters
    :param m: number of candidates
    :param k: number of candidates each voter votes on / reviews per reviewer
    :param l: number of voters each candidate is voted on by
    :return:
    """

    assert n == m
    assert k == l
    assert n == k ** 2

    matrix = np.zeros((n, m), dtype=int)

    ones_per_col = l
    ones_per_row = k

    # Step 1: Assign exactly 'ones_per_row' ones to each row
    for row in range(n):
        ones_positions = np.random.choice(m, ones_per_row, replace=False)
        matrix[row, ones_positions] = 1

    # Step 2: Adjust columns to satisfy the condition
    while True:
        column_sums = matrix.sum(axis=0)
        if all(column_sums == ones_per_col):
            break

        # Find columns that have too many or too few ones
        overfilled_columns = np.where(column_sums > ones_per_row)[0]
        underfilled_columns = np.where(column_sums < ones_per_row)[0]

        for over, under in zip(overfilled_columns, underfilled_columns):
            # Swap a 1 from the overfilled column to the underfilled column
            for row in range(n):
                if matrix[row, over] == 1 and matrix[row, under] == 0:
                    matrix[row, over], matrix[row, under] = 0, 1
                    break

    return matrix


# def generate_assignments(n, m, k, l):
#     """
#     Generate assignment matrix of which voter votes on which candidates.
#     Alternatively, think of as assignments of reviewers to papers.
#     :param n: number of voters
#     :param m: number of candidates
#     :param k: number of candidates each voter votes on / reviews per reviewer
#     :param l: number of voters each candidate is voted on by
#     :return:
#     """
#     while True:
#         # Initialize an n x n matrix filled with zeros
#         matrix = np.zeros((m, n), dtype=int)
#
#         for i in range(m):
#             # Randomly select k distinct columns
#             available = [x for x in range(n) if sum(matrix[:, x]) < l - 1]
#             if len(available) < k:
#                 cols = available + random.sample([x for x in range(n) if sum(matrix[:, x]) == l - 1],
#                                                  k - len(available))
#             else:
#                 cols = random.sample(available, k)
#             for col in cols:
#                 matrix[i][col] = 1
#
#         return matrix


def profile_ranking_from_rule(rule, profile, **kwargs):
    """
    Given a voting rule, convert it to a social welfare function and run that swf on the given profile.
    :param rule: PrefVoting VotingMethod or another function which accepts a profile and returns a ranking
    :param profile: a PrefVoting Profile
    :return: A ranking over the alternatives in the given profile.
    """
    if isinstance(rule, pref_voting.voting_method.VotingMethod):
        swf = pref_voting.helper.swf_from_vm(rule,
                                             tie_breaker=None)  # can also do 'random' or 'alphabetic' tie breaking.
        ranking = swf(profile)
        ranking = ranking.to_indiff_list()
    elif callable(rule):
        if kwargs:
            ranking = rule(profile, **kwargs)
        else:
            ranking = rule(profile)
    else:
        print(f"Given rule is: {rule}")

    return ranking


def kendall_tau_distance(r1, r2, weights=None, normalize=True, rank_map1=None, rank_map2=None):
    """
    Compute the Kendall Tau distance (bubble sort distance) between two tuple rankings.

    Parameters:
    ranking1 (tuple): First ranking as a tuple of tuples
    ranking2 (tuple): Second ranking as a tuple of tuples

    Returns:
    int: Kendall Tau distance between the two rankings
    """

    if rank_map1 is None or rank_map2 is None:
        # Flatten the inner tuples and extract tie information
        if any(isinstance(x, tuple) for x in r1):
            # make list of rankings/tied alternatives
            ranks_with_ties1 = []
            # tied_indices1 = []
            ranking1 = []
            current_rank = 0
            for group in r1:
                for item in group:
                    ranking1.append(item)
                    ranks_with_ties1.append(current_rank)
                    # tied_indices1.append(1 if len(group) > 1 else 0)
                current_rank += len(group)
        # elif isinstance(r1, list):
        #     ranking1 = [r1]
        #     ranks_with_ties1 = [i for i in r1]
        else:
            ranks_with_ties1 = [i for i in range(len(r1))]
            ranking1 = r1

            # ranking1 = tuple(x for inner_tuple in ranking1 for x in inner_tuple)
        if any(isinstance(x, tuple) for x in r2):
            # make list of rankings/tied alternatives
            ranks_with_ties2 = []
            # tied_indices2 = []
            ranking2 = []
            current_rank = 0
            for group in r2:
                for item in group:
                    ranking2.append(item)
                    ranks_with_ties2.append(current_rank)
                    # tied_indices2.append(1 if len(group) > 1 else 0)
                current_rank += len(group)
            # ranking2 = tuple(x for inner_tuple in ranking2 for x in inner_tuple)
        # elif isinstance(r2, list):
        #     ranking2 = [r2]
        #     ranks_with_ties2 = [i for i in r2]
        else:
            ranks_with_ties2 = [i for i in range(len(r2))]
            ranking2 = r2

        # assert len(ranking1) == len(ranking2)
        # m = len(ranking1)
        m = max(max(ranking1), max(ranking2)) + 1

        rank_map1 = dict()
        rank_map2 = dict()
        for i in range(m):
            rank_map1[ranking1[i]] = ranks_with_ties1[i]    # points to all alternatives that are tied for a rank
            rank_map2[ranking2[i]] = ranks_with_ties2[i]
    else:
        m = len(rank_map1)

    if weights is None:
        weights = [1 for _ in range(m)]

    # swaps = 0.0
    # for i in range(m):
    #     for j in range(i + 1, m):
    #
    #         if (rank_map1[i] >= rank_map1[j] and rank_map2[j] >= rank_map2[i]) or (
    #                 rank_map1[j] >= rank_map1[i] and rank_map2[i] >= rank_map2[j]):
    #             if (rank_map1[i] == rank_map1[j]) or (rank_map2[i] == rank_map2[j]):
    #                 # If either candidate is in a tie in either ranking, count that as a half swap
    #                 swaps += 0.5
    #             else:
    #                 # If neither candidate is in a tie, a full swap
    #                 swaps += 1.0

    swaps = 0.0
    for i in range(m):
        for j in range(i+1, m):
            if i == j:
                continue

            weight = weights[i] * weights[j]

            if (rank_map1[i] >= rank_map1[j] and rank_map2[j] >= rank_map2[i]) or (
                    rank_map1[j] >= rank_map1[i] and rank_map2[i] >= rank_map2[j]):
                if (rank_map1[i] == rank_map1[j]) or (rank_map2[i] == rank_map2[j]):
                    # If either candidate is in a tie in either ranking, count that as a half swap
                    swaps += weight * 0.5
                else:
                    # If neither candidate is in a tie, a full swap
                    swaps += weight * 1.0

    if normalize:
        swaps /= ((m ** 2 - m) / 2)

    return swaps


def scores_to_tuple_ranking(scores):
    """
    ChatGPT helper func to convert scores for each alternatives into a tuple ranking where alternatives in same
    tuple are tied with each other.
    :param scores: dict or list. If dict, a mapping of candidate to points. If list, a list of points for the candidate
    in that index.
    :return:
    """
    if isinstance(scores, dict):
        indexed_scores = [(i, scores[i]) for i in range(len(scores))]
    elif isinstance(scores, list):
        indexed_scores = list(enumerate(scores))
    elif isinstance(scores, np.ndarray):
        indexed_scores = list(enumerate(scores.tolist()))
    else:
        raise ValueError(
            f"Unexpected value when converting scores to rankings. Should be dict, list, or ndarray. Is: {scores} of type {type(scores)}")

    # Sort by score (descending) and index (ascending in case of ties)
    indexed_scores.sort(key=lambda x: (-x[1], x[0]))

    # Group tied scores
    result = []
    current_score = None
    current_group = []

    for index, score in indexed_scores:
        if score != current_score:  # New score group
            if current_group:  # Save the previous group
                result.append(tuple(current_group))
            current_group = [index]
            current_score = score
        else:  # Continue the same score group
            current_group.append(index)

    # Append the last group
    if current_group:
        result.append(tuple(current_group))

    return result


# def rule_name(rule):
#     if rule in rule_names:
#         return rule_names[rule]
#     else:
#         raise ValueError(f"Unexpected rule passed. Given: {rule}")


def plurality_ranking_vector(m):
    score_vector = [1] + [0 for _ in range(m - 1)]
    return score_vector


@method_name(name="Plurality", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def plurality_ranking(profile, reverse_vector=False, **kwargs):
    # scores = profile.plurality_scores()
    # ranking = scores_to_tuple_ranking(scores)
    # return ranking
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    score_vector = [1] + [0 for _ in range(k - 1)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def plurality_veto_ranking_vector(m):
    score_vector = [1] + [0.5 for _ in range(m - 2)] + [0]
    return score_vector


@method_name(name="Plurality Veto", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def plurality_veto_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    # score_vector = [1] + [0 for _ in range(k - 2)] + [-1]
    # score_vector = [1] + [0.5 for _ in range(k - 2)] + [0]
    score_vector = plurality_veto_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def borda_ranking_vector(m):
    score_vector = [(m - i - 1) / m for i in range(m)]
    return score_vector


@method_name(name="Borda", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def borda_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    # score_vector = [(k - i - 1)/k for i in range(k)]
    score_vector = borda_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Borda Min-Max", rule_type="positional_scoring", reversible=False)
def borda_minmax_ranking(profile, **kwargs):
    """
    Borda's method but the top and bottom ranking that each alternative receives is removed.
    If multiple voters give the same top/bottom ranking, only one is removed.
    :param profile:
    :param reverse_vector:
    :return:
    """

    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [(k - i - 1)/k for i in range(k)]

    # count number of voters ranking each alternative at each rank
    ranking_counts = [[0 for _ in range(k)] for _ in range(m)]
    # ranking_counts[i][j] is number of voters ranking candidate i in position j
    for ranking in profile:
        for rank, tied_alternatives in enumerate(ranking):
            for alternative in tied_alternatives:
                ranking_counts[alternative][rank] += 1

    scores = [0 for _ in range(m)]
    removed_alternative_ranking_counts = np.zeros(m)    # track how many rankings we remove for each alternative
    for alternative, rc in enumerate(ranking_counts):
        # remove one top and one bottom ranking
        if sum(rc) == 0:
            # skip anyone not ranked in this split
            continue
        nz = np.nonzero(rc)
        first_nz = nz[0][0]     # highest rank this voter received
        last_nz = nz[0][-1]     # lowest rank this voter received

        # if voter was ranked only once, don't subtract too much
        if first_nz == last_nz and rc[first_nz] == 1:
            rc[first_nz] -= 1
            removed_alternative_ranking_counts[alternative] -= 1
        else:
            rc[first_nz] -= 1
            rc[last_nz] -= 1
            removed_alternative_ranking_counts[alternative] -= 2

        # multiply each rank count by appropriate score for its rank
        for rank, count in enumerate(rc):
            scores[alternative] += count * score_vector[rank]

    # count number of times each alternative appears
    # need to subtract from each voter's frequency
    alternative_frequencies = np.zeros(m)
    for alt in range(m):
        for order in profile:
            if alt in order:
                alternative_frequencies[alt] += 1
    alternative_frequencies += removed_alternative_ranking_counts

    scores = normalize_positional_scores(profile, scores, m, frequencies=alternative_frequencies)

    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Trimmed Borda", rule_type="positional_scoring", reversible=False)
def trimmed_borda_ranking(profile, **kwargs):
    """
    Borda's method but the top and bottom ranking that each alternative receives is removed.
    These should be removed before the method is called (before splits are created) and this method just does Borda.
    :param profile:
    :param reverse_vector:
    :return:
    """
    return borda_ranking(profile, reverse_vector=False, **kwargs)


def antiplurality_ranking_vector(m):
    # score_vector = [0 for _ in range(m - 1)] + [-1]
    score_vector = [1 for _ in range(m - 1)] + [0]
    return score_vector


@method_name(name="Veto", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def antiplurality_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    # score_vector = [0 for _ in range(k - 1)] + [-1]
    score_vector = antiplurality_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def dowdall_ranking_vector(m):
    score_vector = [1 / i for i in range(1, m + 1)]
    return score_vector


@method_name(name="Dowdall", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def dowdall_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    # score_vector = [1 / i for i in range(1, k + 1)]
    score_vector = dowdall_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def two_approval_ranking_vector(m):
    score_vector = [1, 1] + [0 for _ in range(m - 2)]
    return score_vector


@method_name(name="Two Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def two_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    # score_vector = [1, 1] + [0 for _ in range(k - 2)]
    score_vector = two_approval_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def three_approval_ranking_vector(m):
    score_vector = [1, 1, 1] + [0 for _ in range(m - 3)]
    return score_vector


@method_name(name="Three Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def three_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    assert m >= 3
    # score_vector = [1, 1, 1] + [0 for _ in range(k - 3)]
    score_vector = three_approval_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def seven_approval_ranking_vector(m):
    score_vector = [1 for _ in range(7)] + [0 for _ in range(m - 7)]
    return score_vector


@method_name(name="Seven Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def seven_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    assert m >= 7
    # score_vector = [1, 1, 1] + [0 for _ in range(k - 3)]
    score_vector = seven_approval_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def eight_approval_ranking_vector(m):
    score_vector = [1 for _ in range(8)] + [0 for _ in range(m - 8)]
    return score_vector


@method_name(name="Eight Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def eight_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    assert m >= 7
    # score_vector = [1, 1, 1] + [0 for _ in range(k - 3)]
    score_vector = eight_approval_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def nine_approval_ranking_vector(m):
    score_vector = [1 for _ in range(9)] + [0 for _ in range(m - 9)]
    return score_vector


@method_name(name="Nine Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def nine_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    assert m >= 7
    # score_vector = [1, 1, 1] + [0 for _ in range(k - 3)]
    score_vector = nine_approval_ranking_vector(k)
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="k-Approval", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
def k_approval_ranking(profile, num_approvals, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True
    score_vector = [1 if i < num_approvals else 0 for i in range(k)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def olympic_medal_count_ranking_vector(profile, m):
    # We need to take in the profile to count how many different medals there are
    # But pretty sure we could actually just do it like 3-approval after other changes
    scores = [0 for _ in range(m)]
    max_num_medals = 0

    weak_ranking = False
    if isinstance(profile[0][0], tuple):
        weak_ranking = True
    if not weak_ranking:
        raise ("Unexpected usage. Should update to allow non-weak rankings for Olympic data.")
    else:
        # Give one point to every alternative in the top three ranks
        for order in profile:
            num_medals = 0
            for rank in range(len(order)):
                for cand in order[rank]:
                    # each of these candidates is tied for the current rank
                    if rank < 3:
                        # scores[cand] += 1
                        num_medals += 1
            max_num_medals = max(num_medals, max_num_medals)
    score_vector = [1] * min(max_num_medals, m) + [0] * (m - max_num_medals)
    return score_vector


@method_name(name="Medal Count", reversible=False, allows_weak_ranking=True)
def olympic_medal_count_ranking(profile, **kwargs):
    """
    Use only in the olympic medal setting. Return the weak ranking where each candidate gets one point per medal,
    regardless of the medal they received.
    :param profile:
    :param kwargs:
    :return:
    """
    m = kwargs["m"]
    k = kwargs["k"]
    # scores = [0 for _ in range(m)]

    weak_ranking = False
    if isinstance(profile[0][0], tuple):
        weak_ranking = True
    if not weak_ranking:
        raise("Unexpected usage. Should update to allow non-weak rankings for Olympic data.")
    else:
        # # Give one point to every alternative in the top three ranks
        # for order in profile:
        #     for rank in range(len(order)):
        #         for cand in order[rank]:
        #             # each of these candidates is tied for the current rank
        #             if rank < 3:
        #                 scores[cand] += 1
        pass

    # ranking = scores_to_tuple_ranking(scores)

    score_vec = olympic_medal_count_ranking_vector(profile, m)
    positional_scores = positional_scoring_scores(profile,
                                                  score_vector=score_vec,
                                                  m=m,
                                                  k=k,
                                                  normalize=False,
                                                  use_mean_score_on_ties=False)
    positional_ranking = scores_to_tuple_ranking(positional_scores)

    return positional_ranking


def olympic_gold_count_ranking_vector(m):

    score_vector = [1000000, 1000, 1] + [0 for _ in range(m - 3)]
    return score_vector


@method_name(name="Leximax", reversible=False, allows_weak_ranking=True)
def olympic_gold_count_ranking(profile, **kwargs):
    """
    Use only in the olympic medal setting. Return the weak ranking where all gold medal winners beat all silver medal
    winners beat all bronze medal winners. Ties among gold medal counts are broken by silver medal count, etc.
    regardless of the medal they received.
    :param profile:
    :param kwargs:
    :return:
    """
    m = kwargs["m"]
    k = kwargs["k"]
    n = len(profile)    # used for scaling number of points
    scores = [0 for _ in range(m)]

    weak_ranking = False
    if isinstance(profile[0][0], tuple):
        weak_ranking = True
    if not weak_ranking:
        raise("Unexpected usage. Should update to allow non-weak rankings for Olympic data.")
    else:
        # # Give one point to every alternative in the top three ranks
        # for order in profile:
        #     for rank in range(len(order)):
        #         for cand in order[rank]:
        #             if rank == 0:
        #                 # gold medal; give huge number of points
        #                 scores[cand] += n * 1000000
        #             elif rank == 1:
        #                 # silver medal; give medium number of points
        #                 scores[cand] += n * 1000
        #             elif rank == 2:
        #                 # bronze medal; give small number of points
        #                 scores[cand] += 1
        pass

    score_vec = olympic_gold_count_ranking_vector(m)
    positional_scores = positional_scoring_scores(profile,
                                                  score_vector=score_vec,
                                                  m=m,
                                                  k=k,
                                                  normalize=False,
                                                  use_mean_score_on_ties=False)
    positional_ranking = scores_to_tuple_ranking(positional_scores)

    # ranking = scores_to_tuple_ranking(scores)
    # return ranking
    return positional_ranking

def f1_1991_ranking_vector(m):
    score_vector = [10, 6, 4, 3, 2, 1]
    if m < len(score_vector):
        score_vector = score_vector[:m]
    score_vector = score_vector + [0 for _ in range(m - len(score_vector))]
    return score_vector

@method_name(name="F1_rule-1991", reversible=False, allows_weak_ranking=True)
def f1_1991_ranking(profile, **kwargs):
    """
    The position scoring rule used in F1 between 1991 and 2002.
    :param profile:
    :param kwargs:
    :return:
    """
    m = kwargs["m"]
    k = kwargs["k"]
    n = len(profile)    # used for scaling number of points
    # score_vector = [10, 6, 4, 3, 2, 1]
    # if m < len(score_vector):
    #     score_vector = score_vector[:m]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True

    # score_vector = score_vector + [0 for _ in range(m-len(score_vector))]
    score_vector = f1_1991_ranking_vector(m)

    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def f1_2003_ranking_vector(m):
    score_vector = [10, 8, 6, 5, 4, 3, 2, 1]
    if m < len(score_vector):
        score_vector = score_vector[:m]
    score_vector = score_vector + [0 for _ in range(m - len(score_vector))]
    return score_vector


@method_name(name="F1_rule-2003", reversible=False, allows_weak_ranking=True)
def f1_2003_ranking(profile, **kwargs):
    """
    The position scoring rule used in F1 between 2003 and 2009.
    :param profile:
    :param kwargs:
    :return:
    """
    m = kwargs["m"]
    k = kwargs["k"]
    n = len(profile)    # used for scaling number of points
    # score_vector = [10, 8, 6, 5, 4, 3, 2, 1]
    # if m < len(score_vector):
    #     score_vector = score_vector[:m]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True

    # score_vector = score_vector + [0 for _ in range(m-len(score_vector))]
    score_vector = f1_2003_ranking_vector(m)

    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


def f1_2010_ranking_vector(m):
    score_vector = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    if m < len(score_vector):
        score_vector = score_vector[:m]
    score_vector = score_vector + [0 for _ in range(m - len(score_vector))]
    return score_vector


@method_name(name="F1_rule-2010", reversible=False, allows_weak_ranking=True)
def f1_2010_ranking(profile, **kwargs):
    """
    The position scoring rule used in F1 between 2010 and 2018.
    :param profile:
    :param kwargs:
    :return:
    """
    m = kwargs["m"]
    k = kwargs["k"]
    n = len(profile)    # used for scaling number of points
    # score_vector = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    # if m < len(score_vector):
    #     score_vector = score_vector[:m]
    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True

    # score_vector = score_vector + [0 for _ in range(m-len(score_vector))]
    score_vector = f1_2010_ranking_vector(m)

    scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=normalize)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Copeland", reversible=False)
def copeland_ranking(profile, **kwargs):
    scores = profile.copeland_scores()
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Random", reversible=False)
def random_ranking(profile, **kwargs):
    # Totally random ranking with no ties as a simple sanity check
    m = kwargs["m"]
    k = kwargs["k"]
    ranking = list(range(k))
    random.shuffle(ranking)
    ranking = [(r,) for r in ranking]
    return ranking


@method_name(name="Annealing Score Vector", reversible=False)
def annealing_ranking(profile, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]

    # initial_state = [1] + [0 for _ in range(k - 1)]
    initial_state = [k-i-1 for i in range(k)]

    if "n_splits" in kwargs:
        n_splits = kwargs["n_splits"]
    else:
        n_splits = 10

    if "n_steps" in kwargs:
        n_steps = kwargs["n_steps"]
    else:
        n_steps = 500

    # profiles = np.array(profile.rankings)
    # tsp = ScoreVectorAnnealer(initial_state=initial_state,
    #                           profiles=[profiles],
    #                           n_splits=n_splits)
    tsp = SingleProfileScoreVectorAnnealer(initial_state=initial_state,
                                           profile=profile,
                                           n_splits=n_splits,
                                           m=m,
                                           k=k)

    tsp.steps = n_steps
    vector, sw = tsp.anneal()
    # print(f"Best Annealing result is: {vector}")

    # s1 = pref_voting.profiles.Profile(profile)
    ranking = positional_scoring_scores(profile, score_vector=vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(ranking)

    return ranking


@method_name(name="Single Profile Annealing", reversible=False)
def annealing_ranking_from_splits(profile, all_s1=None, all_s2=None, return_vector=False, **kwargs):

    if all_s1 is None and all_s2 is None:
        print("Not given splits. Falling back to mildly incorrect annealing method.")
        return annealing_ranking(profile, **kwargs)
    m = kwargs["m"]
    k = kwargs["k"]
    if "use_mean_score_on_ties" in kwargs:
        use_mean_score_on_ties = kwargs["use_mean_score_on_ties"]
    else:
        use_mean_score_on_ties = True

    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True

    if "all_annealing_states" in kwargs:
        all_annealing_states = kwargs["all_annealing_states"]
    else:
        all_annealing_states = [
            [(m - i - 1) / m for i in range(k)]  # If not specified, start only from Borda
        ]

    # initial_state = [1] + [0 for _ in range(k - 1)]   # Plurality
    initial_state = [k - i - 1 for i in range(k)]           # Borda

    if "n_steps" in kwargs:
        n_steps = kwargs["n_steps"]
    else:
        n_steps = 500

    best_energy = 100000000
    best_annealed_vector = all_annealing_states[0]

    for initial_state in all_annealing_states:

        tsp = SingleProfileScoreVectorAnnealer(initial_state=initial_state,
                                               profile=profile,
                                               m=m,
                                               k=k,
                                               all_s1=all_s1,
                                               all_s2=all_s2,
                                               use_mean_score_on_ties=use_mean_score_on_ties,
                                               normalize=normalize)
        # print(f"Initial annealing energy is: {tsp.energy()}")

        tsp.steps = n_steps
        annealed_score_vector, sw = tsp.anneal()

        if sw < best_energy:
            print(f"Found new best annealing vector from start state: {initial_state}")
            best_annealed_vector = annealed_score_vector
            best_energy = sw

    # test_scores = positional_scoring_scores(profile, score_vector=[1, 1, 1], m=m, k=k)
    # ranking_test = scores_to_tuple_ranking(test_scores)
    ranking_scores = positional_scoring_scores(profile,
                                               score_vector=best_annealed_vector,
                                               m=m,
                                               k=k,
                                               use_mean_score_on_ties=use_mean_score_on_ties
                                               )
    ranking = scores_to_tuple_ranking(ranking_scores)
    # prett_ranking_scores = positional_scoring_scores(profile, score_vector=prettify_positional_scores(best_annealed_vector), m=m, k=k)
    # ranking_pretty = scores_to_tuple_ranking(prett_ranking_scores)

    if return_vector:
        return ranking, best_annealed_vector
    else:
        return ranking


@method_name(name="Gradient Optimized Scores", reversible=False)
def gradient_ranking(profile, **kwargs):
    # Example function: a simple sigmoid
    # (n, m, k, l) = args
    m = kwargs["m"]
    k = kwargs["k"]
    n = len(profile)

    verbose = True
    n_splits = 1
    for _ in range(n_splits):
        try:
            s1, s2 = rule_comparison.split_data(profile, n=n, m=m)
            proposal_split = np.zeros((m, 2))
            for ranking in s1:
                for cand in ranking:
                    proposal_split[cand][0] += 1
            for ranking in s2:
                for cand in ranking:
                    proposal_split[cand][1] += 1

            proposal_split = np.min(proposal_split, 1)
            gamma = 2
            l = k
            weights = ((gamma ** proposal_split) - 1) / (gamma ** (int(l / 2)) - 1)

            losses = []
            # a_initial_values = torch.cat((torch.ones(k // 2), torch.zeros(k-(k//2))))
            a_initial_values = torch.linspace(1.0, 0.0, steps=k)
            score_vector = [torch.tensor([val], requires_grad=True) for val in a_initial_values]  # check different initializiations

            # # Optimizer: here we use basic gradient descent with a learning rate of 1e-1
            # if opt_type == "Adam":
            #     optimizer = torch.optim.Adam([a[i] for i in range(1, k - 1)], lr=lr)
            # elif opt_type == "SGD":
            #     optimizer = torch.optim.SGD([a[i] for i in range(1, k - 1)], lr=lr)
            optimizer = torch.optim.SGD([score_vector[i] for i in range(1, k - 1)], lr=0.1)

            # Number of steps we want to take

            # Gradient descent loop
            torch.autograd.set_detect_anomaly(False)
            min_steps = 100
            max_steps = 1000
            sensitivity = 0.004
            step = 0
            steep = 300
            old_loss = 100001
            new_loss = 100000
            good_runs = 0
            n_plat = 0
            max_n_plateau = 50
            lr = 0.01
            a_values = [ai.item() for ai in score_vector]
            if verbose:
                print(f"Step {step}: a = {a_values}")

            while (lr > 1e-6) and (n_plat < max_n_plateau or step < min_steps) and step < max_steps:
                if old_loss - new_loss <= sensitivity:  # if the loss is not decreasing for too many steps, quit
                    n_plat += 1
                else:
                    n_plat = 0
                if new_loss > old_loss + 0.1 / (step // 100 + 1):
                    lr = lr / 10
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    if verbose:
                        print("NEW LEARNING RATE:", lr)
                    good_runs = 0
                elif new_loss < old_loss:
                    good_runs += 1
                    if good_runs > 100:
                        lr = lr * 10
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        if verbose:
                            print("NEW LEARNING RATE:", lr)
                        good_runs = 0
                else:
                    good_runs = 0

                # Zero the gradients from the previous step
                old_loss = new_loss
                optimizer.zero_grad()
                # Calculate the score for each split
                # score for each alternative under the current scoring vector
                # total_score_1, total_score_2 = total_scores(M, score_vector, args)
                total_score_1 = positional_scoring_scores(s1, score_vector, m, k, normalize=False)
                total_score_2 = positional_scoring_scores(s2, score_vector, m, k, normalize=False)
                # M_sum = M[2]
                # total_score = [sum(a[i] * M_sum[:, p][i] for i in range(k)) for p in range(m)]
                d = [[0] * m for _ in range(m)]  # to host the errors
                for i in range(m):
                    for j in range(i + 1, m):
                        d[i][j] = weights[i] * weights[j] * torch.special.expit(
                            -(total_score_1[i] - total_score_1[j]) * (total_score_2[i] - total_score_2[j]) * steep)

                y = sum([d[i][j] for i in range(m) for j in range(i + 1, m)])

                if step == 0:
                    y_best = y
                    # a_best = [ai.item() for ai in score_vector]
                if y < y_best:
                    y_best = torch.clone(y)
                    # a_best = [ai.item() for ai in score_vector]

                # Compute gradients of the function with respect to x
                y.backward()

                # Take a step in the direction of the negative gradient
                optimizer.step()
                # Project the variables to satisfy the constraints a[i] <= a[i-1] for all i
                with torch.no_grad():
                    for i in range(1, k - 1):
                        score_vector[i].data = torch.minimum(score_vector[i], score_vector[i - 1])
                        score_vector[i].data = torch.maximum(score_vector[i], score_vector[k - 1])

                new_loss = y.item()
                losses.append(new_loss)

                if step % 10 == 0 and verbose:
                    # Print the current value of a and the function value
                    # a_values = [ai.item() for ai in score_vector]
                    # print(f"Step {step}: sum of sigmoids = {y.item()}")
                    print(f"Step {step} loss is: {losses[-1]}")

                    # print(f"Step {step}: a = {a_values}, sum of sigmoids = {y.item()}")
                    # for i in range(k):
                    #     print("gradient of a[",i,"] is: ", score_vector[i].grad)
                step += 1

            scores = positional_scoring_scores(profile, score_vector, m=m, k=k, normalize=False)
            ranking = scores_to_tuple_ranking(scores)

            return ranking

            #
            # The variable x should now be at the minimum of the sigmoid function
            # which will be close to the point where the derivative is zero.

        except AttributeError:
            print('Encountered an attribute error')


def instant_runoff_with_incomplete_ballots(profile):
    """
    Compute the full ranking based on an instant runoff vote from a list of partial preference rankings,
    supporting ties within individual voter preferences.

    Args:
        rankings: A list of lists or lists of sets, where each inner list represents a voter's ranked preferences.
                 Each element in the inner list is either:
                 - A single alternative (no tie)
                 - A set of alternatives (tied at this preference level)

                 Example with ties: [['A', {'B', 'C'}, 'D'], [{'A', 'B'}, 'C'], [{'C', 'D'}, 'A']]
                 In the first voter's ranking, B and C are tied for second place.

    Returns:
        A list representing the full ranking of alternatives, where:
        - The first element is the winner (last alternative remaining)
        - The last element is the first alternative eliminated
        - Returns an empty list if there are no valid votes
    """
    # # Normalize the rankings to ensure sets for tied alternatives
    # normalized_rankings = []
    # for ranking in rankings:
    #     normalized_ranking = []
    #     for pref in ranking:
    #         if isinstance(pref, tuple):
    #             normalized_ranking.append(pref)
    #         else:
    #             normalized_ranking.append({pref})  # Convert single alternatives to singleton sets
    #     normalized_rankings.append(normalized_ranking)

    # # Filter out empty rankings
    # valid_rankings = [ranking for ranking in normalized_rankings if ranking]

    # if not valid_rankings:
    #     return []
    valid_rankings = profile

    # Get the set of all alternatives from the rankings
    all_alternatives = set()
    for ranking in profile:
        for tied_alts in ranking:
            all_alternatives.update(tied_alts)

    # Track elimination order
    elimination_order = []

    # Continue eliminating alternatives until we have a winner
    remaining_alternatives = set(all_alternatives)

    while len(remaining_alternatives) > 1:
        # Count first preferences for each remaining alternative, accounting for ties
        counts = {alt: 0 for alt in remaining_alternatives}

        for ranking in valid_rankings:
            # Find all alternatives at the highest remaining rank
            top_alternatives = set()
            for pref_set in ranking:
                # Get the alternatives from this preference level that are still in the running
                remaining_in_pref = {alt for alt in pref_set if alt in remaining_alternatives}
                if remaining_in_pref:
                    top_alternatives = remaining_in_pref
                    break

            # Distribute one vote equally among all top alternatives
            if top_alternatives:
                vote_fraction = 1.0 / len(top_alternatives)
                for alt in top_alternatives:
                    counts[alt] += vote_fraction

        # If there are no valid votes (all preferences have been eliminated)
        if sum(counts.values()) == 0:
            # Add remaining alternatives to elimination order (tie for last place)
            sorted_remaining = sorted(remaining_alternatives)  # Sort for deterministic output
            elimination_order.extend(sorted_remaining)
            remaining_alternatives.clear()
            break

        # Find the alternative(s) with the minimum count
        min_count = min(counts.values())
        alternatives_to_eliminate = [alt for alt, count in counts.items() if count == min_count]

        # Check if there's a tie for first place (everyone has same count)
        if min_count == max(counts.values()):
            # In case of a tie for first, use tie-breaker to determine elimination order
            elimination_order.extend(reverse_tie_breaker(alternatives_to_eliminate, valid_rankings))
            break

        # Handle the case of multiple alternatives with the minimum count
        # Sort for deterministic results when multiple alternatives share the same count
        alternatives_to_eliminate.sort()

        # Add the eliminated alternatives to our elimination order
        elimination_order.extend(alternatives_to_eliminate)

        # Eliminate the alternative(s) with the minimum count
        for alt in alternatives_to_eliminate:
            remaining_alternatives.remove(alt)

    # Add the winner (last remaining alternative) to the elimination order
    if remaining_alternatives and len(elimination_order) < len(all_alternatives):
        elimination_order.extend(sorted(remaining_alternatives))

    # Return the reversed elimination order (winner first, first eliminated last)
    return list(reversed(elimination_order))


def reverse_tie_breaker(tied_alternatives, rankings):
    """
    Breaks ties and returns the alternatives in order (from best to worst).

    Args:
        tied_alternatives: List of alternatives that are tied
        rankings: List of voter preference rankings (normalized with sets)

    Returns:
        Sorted list of tied alternatives (from best to worst)
    """
    # Count how many times each alternative appears at the top among tied alternatives
    appearances = {alt: 0 for alt in tied_alternatives}

    for ranking in rankings:
        top_tied = set()
        for pref_set in ranking:
            # Find tied alternatives in this preference level
            tied_in_pref = {alt for alt in pref_set if alt in tied_alternatives}
            if tied_in_pref:
                top_tied = tied_in_pref
                break

        # Distribute one vote equally among all top tied alternatives
        if top_tied:
            vote_fraction = 1.0 / len(top_tied)
            for alt in top_tied:
                appearances[alt] += vote_fraction

    # Sort alternatives by number of appearances (highest to lowest)
    sorted_alternatives = sorted(tied_alternatives, key=lambda alt: (-appearances[alt], alt))
    return sorted_alternatives


@method_name(name="PL MLE", reversible=False)
def choix_pl_ranking(profile, **kwargs):
    m = kwargs["m"]

    # profile = profile.rankings
    ranking = choix.opt_rankings(m, profile)
    ranking = scores_to_tuple_ranking(ranking)
    return ranking


def positional_scoring_scores(profile, score_vector, m, k, normalize=True, use_mean_score_on_ties=True):
    """
    TODO: Could probably be made way faster with numpy.
    :param profile:
    :param score_vector:
    :param m: number of alternatives
    :param k: number of alternatives ranked by each voter
    :return:
    """
    # assert len(score_vector) == len(profile.rankings[0])

    # m = len(score_vector)
    scores = [0 for _ in range(m)]

    if len(profile) == 0:
        # no orders have been assigned to the profile; all candidates should be tied
        return scores

    weak_ranking = False
    if isinstance(profile[0][0], tuple):
        weak_ranking = True

    if not weak_ranking:
        for order in profile:
            for rank, o in enumerate(order):
                scores[o] += score_vector[rank]
    else:
        # points given to each alternative are the average of all positions occupied by tied alternatives
        for order in profile:
            curr_idx = 0
            for tied_alternatives in order:
                if len(tied_alternatives) == 0:
                    continue

                num_tied = len(tied_alternatives)
                if use_mean_score_on_ties:
                    points_per_alternative = np.mean(score_vector[curr_idx:curr_idx+num_tied])
                    curr_idx += num_tied
                else:
                    points_per_alternative = score_vector[curr_idx]
                    curr_idx += 1

                for tied_alt in tied_alternatives:
                    scores[tied_alt] += points_per_alternative


    if normalize:
        scores = normalize_positional_scores(profile, scores, m)
    else:
        # print("breakpoint plz")
        pass

    return scores


def normalize_positional_scores(profile, scores, m, frequencies=None):
    """
    NOTE: This is not "standard" normalization to make things less than one, but rather is normalization to ensure
    each position is weighted appropriately for how often each candidate is ranked.
    :param profile:
    :param scores:
    :param m:
    :param frequencies:
    :return:
    """

    if frequencies is None:
        # count number of times each alternative appears
        alternative_frequencies = np.zeros(m)

        weak_ranking = False
        if isinstance(profile[0][0], tuple):
            weak_ranking = True

        if not weak_ranking:
            for alt in range(m):
                for order in profile:
                    if alt in order:
                        alternative_frequencies[alt] += 1
        else:
            for alt in range(m):
                for weak_order in profile:
                    for tied_cands in weak_order:
                        if alt in tied_cands:
                            alternative_frequencies[alt] += 1
    else:
        alternative_frequencies = frequencies

    # normalize score vector by the number of times each alternative appears in the profile
    scores = np.array(scores)
    scores = np.divide(scores, alternative_frequencies, out=np.zeros_like(scores, dtype=float),
                       where=scores != 0).tolist()

    return scores


def prettify_positional_scores(scores, rounding=False):
    """
    The "standard" normalization -- ensure that all values are between zero and one in a way that won't change the
    ranking induced by the scores.
    Subtract the min score from each value then divide by the max score.
    :param scores: A list corresponding to a positional scoring vector.
    :param rounding: bool or int. If int, the number of digits that should be rounded to. If True, arbitrarily use 3 digits
    :return:
    """
    if len(set(scores)) == 1:
        return [1.0] * len(scores)
    min_score = min(scores)
    scores = [s-min_score for s in scores]
    max_score = max(scores)
    scores = [s/max_score for s in scores]
    if rounding:
        if isinstance(rounding, bool):
            round_digits = 3
        elif isinstance(rounding, int):
            round_digits = rounding
        else:
            raise ValueError("Need to pass bool or int to rounding.")
        scores = [round(s, round_digits) for s in scores]
    return scores


@method_name(name="Kemeny", reversible=False)
def kemeny_gurobi(profile, time_out=None, printout_mode=False, **kwargs):
    """
    Kemeny-Young optimal rank aggregation.
    :param profile: Preference profile containing the complete ranking of each voter.
    :param args:
    :param time_out:
    :param printout_mode:
    :return:
    """
    n = len(profile)
    m = kwargs["m"]
    k = len(profile[0])
    l = k
    assignments = generate_assignments(n, m, k, l)



@method_name(name="Kemeny", reversible=False)
def kemeny_gurobi_lazy(profile, time_out=None, printout_mode=True, **kwargs):
    """Kemeny-Young optimal rank aggregation"""
    # (n, m, k, l) = args
    # profile = profile.rankings
    n = len(profile)
    m = kwargs["m"]
    k = len(profile[0])
    l = k

    # maximize c.T * x
    edge_weights = build_pairwise_graph(profile, (n, m, k, l))
    edge_weights_np = edge_weights.numpy()

    G = nx.DiGraph()

    for i in range(m):
        for j in range(m):
            if edge_weights_np[i, j] != 0:  # assuming 0 means no edge
                G.add_edge(i, j, weight=edge_weights_np[i, j])
    G = gpcm.add_orig_edges_map(G)
    G = nx.DiGraph(G)
    # time_out = 150
    # t1 = time.time()
    # print(f"Starting Kemeny with time_out={time_out}")
    elims, cost, cycle_matrix = gl.solve_problem(G, time_out=time_out, print_mode=printout_mode)
    # print(f"Just finished Kemeny. Took {time.time() - t1}")
    for (u, v) in elims:
        edge_weights_np[u, v] = 0.0
    ranking = topological_sort_kahn(edge_weights_np)
    # print(ranking)
    # scores=np.argsort(ranking)
    # print(scores)
    # for i in range(m):
    #     for j in range(i+1,m):
    #         if edge_weights_np[i,j]>0:
    #             assert scores[i] < scores[j]

    ranking = [(i,) for i in ranking]

    return ranking
    # return ranking, cost


def build_pairwise_graph(reviews, args):  # given reviews (pytorch tensor), generates the pairwise comparison matrix
    # Initialize tensor B with zeros
    (n, m, k, l) = args
    B = torch.zeros(m, m, dtype=torch.int32)
    for row in reviews:
        for i, first_num in enumerate(row):
            for second_num in row[i + 1:]:
                B[first_num, second_num] += 1
                B[second_num, first_num] -= 1
    return torch.maximum(B, torch.tensor(0.))


def topological_sort_kahn(adj_matrix):  # topologically sorts a DAG, ChatGPT generated.
    n = adj_matrix.shape[0]
    in_degree = np.count_nonzero(adj_matrix, axis=0)
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []

    while queue:
        node = queue.pop(0)
        top_order.append(node)
        for i in range(n):
            if adj_matrix[node, i] > 0:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(top_order) != n:
        return None  # Cycle detected or graph is not a DAG
    return top_order


def reviewer_split(reviews, reviewers1, reviewers2, n, m, k, l, gamma):
    """
    Splits the reviewer set into two and returns a M^(k) matrix for each of k in {1,2}, where M^(k)_{ij} records how many reviewers ranked proposal i in position j
    Splits set of voters into two groups.
    Creates a ranking matrix for each different split.
    :param reviews:
    :param args:
    :param gamma:
    :return:
    """
    # (n, m, k, l) = args
    all_reviewers = [i for i in range(n)]
    # reviewers1 = set(random.sample(all_reviewers, int(n / 2)))
    # reviewers2 = set(all_reviewers) - reviewers1
    M1 = torch.Tensor([[0] * m for _ in range(k)])  # M1_jp = how many reviewers in set1 ranked prporsal p on position j
    M2 = torch.Tensor([[0] * m for _ in range(k)])  # M2_jp = how many reviewers in set2 ranked prporsal p on position j
    M_sum = torch.Tensor([[0] * m for _ in range(k)])  # in total
    proposal_split = np.zeros((m, 2))
    reviewers = [reviewers1, reviewers2]
    M = [M1, M2, M_sum]

    for i in [0, 1]:
        for r in reviewers[i]:
            for j in range(len(reviews[r])):
                M[i][j][reviews[r][j]] += 1
                M[2][j][reviews[r][j]] += 1
                proposal_split[reviews[r][j]][i] += 1
        total_reviewers = torch.sum(M[i], dim=0)
        mask = total_reviewers != 0
        M[i][:, mask] = M[i][:, mask] / total_reviewers[mask]
    total_reviewers = torch.sum(M[2], dim=0)
    mask = total_reviewers != 0
    M[2][:, mask] = M[2][:, mask] / total_reviewers[mask]

    proposal_split = np.min(proposal_split, 1)
    weights = ((gamma ** proposal_split) - 1) / (gamma ** (int(l / 2)) - 1)
    return reviewers, M, weights


def _weight_of_splits(m, l, s1, s2, giving_rank_counts=False):
    gamma = 2
    # calculate weight based on how balanced each ranking is
    proposal_split = np.zeros((m, 2))
    if not giving_rank_counts:
        for ranking in s1:
            if isinstance(ranking[0], tuple):
                # in a weak ranking
                for tied_cands in ranking:
                    for cand in tied_cands:
                        proposal_split[cand][0] += 1
            else:
                for cand in ranking:
                    proposal_split[cand][0] += 1    # count number of times each candidate is in split 1
        for ranking in s2:
            if isinstance(ranking[0], tuple):
                # in a weak ranking
                for tied_cands in ranking:
                    for cand in tied_cands:
                        proposal_split[cand][1] += 1
            else:
                for cand in ranking:
                    proposal_split[cand][1] += 1    # count number of times each candidate is in split 1
    if giving_rank_counts:
        for cand, cand_ranks in enumerate(s1):
            proposal_split[cand, 0] += sum(cand_ranks)
        for cand, cand_ranks in enumerate(s2):
            proposal_split[cand, 1] += sum(cand_ranks)


    proposal_split = np.min(proposal_split, 1)
    weights = ((gamma ** proposal_split) - 1) / (gamma ** (int(l / 2)) - 1)
    return weights