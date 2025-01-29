import pref_voting
import torch
import numpy as np
import random
import choix
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


def kendall_tau_distance(r1, r2, weights=None, normalize=True):
    """
    Compute the Kendall Tau distance (bubble sort distance) between two tuple rankings.

    Parameters:
    ranking1 (tuple): First ranking as a tuple of tuples
    ranking2 (tuple): Second ranking as a tuple of tuples

    Returns:
    int: Kendall Tau distance between the two rankings
    """
    # Flatten the inner tuples and extract tie information
    if any(isinstance(x, tuple) for x in r1):
        # make list of rankings/tied alternatives
        ranks_with_ties1 = []
        tied_indices1 = []
        ranking1 = []
        current_rank = 0
        for group in r1:
            for item in group:
                ranking1.append(item)
                ranks_with_ties1.append(current_rank)
                tied_indices1.append(1 if len(group) > 1 else 0)
            current_rank += len(group)
    # elif isinstance(r1, list):
    #     ranking1 = [r1]
    #     ranks_with_ties1 = [i for i in r1]
    else:
        ranks_with_ties1 = [i for i in range(r1)]
        tied_indices1 = [0 for _ in range(r1)]
        ranking1 = [r1]

        # ranking1 = tuple(x for inner_tuple in ranking1 for x in inner_tuple)
    if any(isinstance(x, tuple) for x in r2):
        # make list of rankings/tied alternatives
        ranks_with_ties2 = []
        tied_indices2 = []
        ranking2 = []
        current_rank = 0
        for group in r2:
            for item in group:
                ranking2.append(item)
                ranks_with_ties2.append(current_rank)
                tied_indices2.append(1 if len(group) > 1 else 0)
            current_rank += len(group)
        # ranking2 = tuple(x for inner_tuple in ranking2 for x in inner_tuple)
    # elif isinstance(r2, list):
    #     ranking2 = [r2]
    #     ranks_with_ties2 = [i for i in r2]
    else:
        ranks_with_ties2 = [i for i in range(r2)]
        tied_indices2 = [0 for _ in range(r2)]
        ranking2 = [r2]

    assert len(ranking1) == len(ranking2)
    m = len(ranking1)

    rank_map1 = dict()
    rank_map2 = dict()
    for i in range(m):
        rank_map1[ranking1[i]] = ranks_with_ties1[i]
        rank_map2[ranking2[i]] = ranks_with_ties2[i]

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


@method_name(name="Plurality", rule_type="positional_scoring", reversible=True)
def plurality_ranking(profile, reverse_vector=False, **kwargs):
    # scores = profile.plurality_scores()
    # ranking = scores_to_tuple_ranking(scores)
    # return ranking
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [1] + [0 for _ in range(k - 1)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Plurality Veto", rule_type="positional_scoring", reversible=True)
def plurality_veto_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [1] + [0 for _ in range(k - 2)] + [-1]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Borda", rule_type="positional_scoring", reversible=True)
def borda_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [(k - i - 1)/k for i in range(k)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
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
        for rank, alternative in enumerate(ranking):
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


@method_name(name="Anti-Plurality", rule_type="positional_scoring", reversible=True)
def antiplurality_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [0 for _ in range(k - 1)] + [-1]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Dowdall", rule_type="positional_scoring", reversible=True)
def dowdall_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [1 / i for i in range(1, k + 1)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="Two Approval", rule_type="positional_scoring", reversible=True)
def two_approval_ranking(profile, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [1, 1] + [0 for _ in range(k - 2)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


@method_name(name="k-Approval", rule_type="positional_scoring", reversible=True)
def k_approval_ranking(profile, num_approvals, reverse_vector=False, **kwargs):
    m = kwargs["m"]
    k = kwargs["k"]
    score_vector = [1 if i < num_approvals else 0 for i in range(k)]
    if reverse_vector:
        score_vector = list(reversed(score_vector))
    scores = positional_scoring_scores(profile, score_vector)
    ranking = scores_to_tuple_ranking(scores)
    return ranking


# two_approval = lambda profile: k_approval_ranking(profile, 2)
# three_approval = lambda profile: k_approval_ranking(profile, 3)
# four_approval = lambda profile: k_approval_ranking(profile, 4)
# five_approval = lambda profile: k_approval_ranking(profile, 5)
# six_approval = lambda profile: k_approval_ranking(profile, 6)


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

    initial_state = [1] + [0 for _ in range(k - 1)]

    if "n_splits" in kwargs:
        n_splits = kwargs["n_splits"]
    else:
        n_splits = 20

    if "n_steps" in kwargs:
        n_steps = kwargs["n_steps"]
    else:
        n_steps = 300

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

    # s1 = pref_voting.profiles.Profile(profile)
    ranking = positional_scoring_scores(profile, score_vector=vector, m=m, k=k)
    ranking = scores_to_tuple_ranking(ranking)

    return ranking


@method_name(name="PL MLE", reversible=False)
def choix_pl_ranking(profile, **kwargs):
    m = kwargs["m"]

    # profile = profile.rankings
    ranking = choix.opt_rankings(m, profile)
    ranking = scores_to_tuple_ranking(ranking)
    return ranking


def positional_scoring_scores(profile, score_vector, m, k):
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

    # profile = profile.rankings

    for order in profile:
        for rank, o in enumerate(order):
            scores[o] += score_vector[rank]

    scores = normalize_positional_scores(profile, scores, m)

    return scores


def normalize_positional_scores(profile, scores, m, frequencies=None):

    if frequencies is None:
        # count number of times each alternative appears
        alternative_frequencies = np.zeros(m)
        for alt in range(m):
            for order in profile:
                if alt in order:
                    alternative_frequencies[alt] += 1
    else:
        alternative_frequencies = frequencies

    # normalize score vector by the number of times each alternative appears in the profile
    scores = np.array(scores)
    scores = np.divide(scores, alternative_frequencies, out=np.zeros_like(scores, dtype=float),
                       where=scores != 0).tolist()

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
def kemeny_gurobi_lazy(profile, time_out=None, printout_mode=False, **kwargs):
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
    elims, cost, cycle_matrix = gl.solve_problem(G, time_out=time_out, print_mode=printout_mode)
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