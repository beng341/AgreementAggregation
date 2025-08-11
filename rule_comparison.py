import itertools
import os
import pprint
import random
import numpy as np
import pref_voting.profiles
import scipy.spatial.distance

from utils import data_utils as du
from utils import voting_utils as vu
import pandas as pd


def make_split_indices(rankings, n_splits, split_type="equal_prob"):
    """
    Make list of pairs, where each pair contains two lists of indices. Each list of indices corresponds to the voters
    in that split. Used to ensure consistent splits when comparing different rules.
    :param rankings:
    :param n_splits:
    :param split_type:
    :return:
    """
    n = len(rankings)
    m = len(rankings[0])
    all_reviewers = [i for i in range(n)]

    all_splits = []

    for _ in range(n_splits):
        # split rankings into two groups
        if split_type == "equal_size":
            reviewers1 = set(random.sample(all_reviewers, int(n / 2)))
            reviewers2 = set(all_reviewers) - reviewers1
            all_splits.append((list(reviewers1), list(reviewers2)))
            # s1, s2 = [rankings[r1] for r1 in reviewers1], [rankings[r2] for r2 in reviewers2]
        elif split_type == "equal_prob":
            splits = np.random.choice([1, 2], size=len(all_reviewers))
            reviewers1 = np.array(all_reviewers)[splits == 1].tolist()
            reviewers2 = np.array(all_reviewers)[splits == 2].tolist()
            # if reviewers1 == []:
            #     reviewers1 = tuple(i for i in range(m))
            # if reviewers2 == []:
            #     reviewers2 = tuple(i for i in range(m))
            all_splits.append((reviewers1, reviewers2))
        else:
            raise ValueError(f"Unexpected value for split_type: {split_type}")
    return all_splits


def splits_from_split_indices(rankings, split_indices):
    """
    Given a complete list of rankings/voters, return the two lists of rankings corresponding to the given split indices.
    :param rankings:
    :param split_indices: A tuple containing two lists of indices which partition the rankings into two sets.
    :return:
    """
    return rankings[split_indices[0]], rankings[split_indices[1]]


def splits_from_split_indices_no_numpy(rankings, split_indices):
    """
    Given a complete list of rankings/voters, return the two lists of rankings corresponding to the given split indices.
    :param rankings:
    :param split_indices: A tuple containing two lists of indices which partition the rankings into two sets.
    :return:
    """
    r0 = [rankings[idx] for idx in split_indices[0]]
    r1 = [rankings[idx] for idx in split_indices[1]]
    return r0, r1


def split_data(rankings, n, m, split_type="equal_size"):
    """
    Split some given set of preference orders (a.k.a. reviews over papers) into two sets of preference orders.
    Assume for now that we have complete rankings and return the unweighted splits.
    Each split is a list of preference orders, each order representing one voter/reviewer.
    :param rankings:
    :param n: n voters
    :param m: m alternatives
    :param split_type: "equal_size" to make two splits of equal size; "equal_prob" to assign each voter to a split with
    equal probability.
    :return:
    """
    all_reviewers = [i for i in range(n)]

    # split rankings into two groups
    if split_type == "equal_size":
        reviewers1 = set(random.sample(all_reviewers, int(n / 2)))
        reviewers2 = set(all_reviewers) - reviewers1
        s1, s2 = [rankings[r1] for r1 in reviewers1], [rankings[r2] for r2 in reviewers2]
    elif split_type == "equal_prob":
        splits = np.random.choice([1, 2], size=len(all_reviewers))
        reviewers1 = np.array(all_reviewers)[splits == 1]
        # reviewers1 = set(reviewers1.tolist())
        reviewers2 = np.array(all_reviewers)[splits == 2]
        # reviewers2 = set(reviewers2.tolist())
        s1, s2 = [rankings[r1] for r1 in reviewers1], [rankings[r2] for r2 in reviewers2]
        if s1 == []:
            s1 = [[tuple(i for i in range(m))]]
        if s2 == []:
            s2 = [[tuple(i for i in range(m))]]
    else:
        raise ValueError(f"Unexpected value for split_type: {split_type}")
    # reviewers = [reviewers1, reviewers2]

    # M, proposal_split = du.count_rankings_in_splits(s1=s1, s2=s2, m=m)

    # M1 = np.zeros((m, m))   # M1_jp = how many reviewers in set1 ranked proposal p on position j
    # M2 = np.zeros((m, m))   # M2_jp = how many reviewers in set2 ranked proposal p on position j
    # M_sum = np.zeros((m, m))   # total rankings of proposal p on position j
    # # M_sum = torch.Tensor([[0] * m for _ in range(k)])  # in total
    # proposal_split = np.zeros((m, 2))
    # reviewers = [reviewers1, reviewers2]
    # M = [M1, M2, M_sum]
    #
    # # Count number preferring r to j in each ranking and in total
    # for i in [0, 1]:
    #     for r in reviewers[i]:
    #         for j in range(len(rankings[r])):
    #             M[i][j][rankings[r][j]] += 1    # current split
    #             M[2][j][rankings[r][j]] += 1    # total count
    #             proposal_split[rankings[r][j]][i] += 1  # number of times proposal j appears in current split (?)
    #     # total_reviewers = torch.sum(M[i], dim=0)
    #     total_reviewers = np.sum(M[i], axis=0)
    #     mask = total_reviewers != 0
    #     M[i][:, mask] = M[i][:, mask] / total_reviewers[mask]
    # total_reviewers = torch.sum(M[2], dim=0)
    # total_reviewers = np.sum(M[2], axis=0)
    # mask = total_reviewers != 0
    # M[2][:, mask] = M[2][:, mask] / total_reviewers[mask]
    #
    # proposal_split = np.min(proposal_split, 1)  # min number of times each alternatives appears
    # weights = ((gamma ** proposal_split) - 1) / ((gamma ** total_reviewers) - 1)
    # weights = ((gamma ** proposal_split) - 1) / (gamma ** (int(l / 2)) - 1)
    # return s1, s2, M, weights
    return s1, s2


def jaccard_distance_between_rankings(r1, r2, weights=None, num_winners=None):
    """
    Quick test function to experiment with the Jaccard Index as a distance metric for a multi-winner setup.
    Number of winners is not really a concept here yet so we hardcode it as suits our experiments.
    :param r1:
    :param r2:
    :param weights:
    :param num_winners:
    :return:
    """

    # convert the top num_winners from each ranking into a binary set
    flat_r1 = [tied_alt for order in r1 for tied_alt in order]
    flat_r2 = [tied_alt for order in r2 for tied_alt in order]
    assert max(flat_r1) == max(flat_r2)     # sanity check

    if num_winners is None and len(flat_r1) < 240:
        num_winners = len(flat_r1) // 2
    elif num_winners is None:
        num_winners = 240

    num_candidates = max(flat_r1)+1
    # winners1 = flat_r1[:num_winners]
    # winners2 = flat_r2[:num_winners]

    winners1 = []
    for idx, order in enumerate(r1):
        order_list = list(order)
        if len(winners1) + len(order) > num_winners:
            print(f"Split 1 at index {idx} overflowing list of ties has length {len(order)}. We take {num_winners-len(winners1)}.")
            # shuffle tied winners and take random selection to get required number of winners
            random.shuffle(order_list)
            winners1 += order_list[:num_winners-len(winners1)]
            break
        elif len(winners1) + len(order) <= num_winners:
            winners1 += order_list

    winners2 = []
    for idx, order in enumerate(r2):
        order_list = list(order)
        if len(winners2) + len(order) > num_winners:
            print(f"Split 2 at index {idx} overflowing list of ties has length {len(order)}. We take {num_winners-len(winners2)}.")
            # shuffle tied winners and take random selection to get required number of winners
            random.shuffle(order_list)
            winners2 += order_list[:num_winners-len(winners2)]
            break
        elif len(winners2) + len(order) <= num_winners:
            winners2 += order_list

    win_set1 = [0] * num_candidates
    for w in winners1:
        win_set1[w] = 1
    win_set2 = [0] * num_candidates
    for w in winners2:
        win_set2[w] = 1

    # compute jaccard similarity
    dist = scipy.spatial.distance.jaccard(win_set1, win_set2, w=weights)
    # dist = scipy.spatial.distance.jaccard(win_set1, win_set2)
    return dist
    # return 1-dist


def kt_distance_between_rankings(r1, r2, weights=None, rank_map1=None, rank_map2=None):
    """
    Find the Kendall-Tau distance between the two given rankings.
    :param r1:
    :param r2:
    :param weights: Unsupported. Do not use.
    :return:
    """
    dist = vu.kendall_tau_distance(r1, r2, weights=weights, rank_map1=rank_map1, rank_map2=rank_map2)
    return dist


def kt_distance_one_profile_one_rule(profile, n_splits, rule, m=None, **kwargs):
    """
    Measure the KT distance between splits on one given profile using a single rule.
    :param profile:
    :param n_splits:
    :param rule:
    :return:
    """
    all_distances = []
    n = len(profile)
    if m is None:
        m = [num for order in profile for num in order]
        m = max(m) + 1
    for _ in range(n_splits):
        s1, s2 = split_data(profile, n=n, m=m)

        kwargs = {"m": m, "k": 6}

        ranking1 = vu.profile_ranking_from_rule(rule, s1, **kwargs)
        ranking2 = vu.profile_ranking_from_rule(rule, s2, **kwargs)

        gamma = 2
        # # calculate weight based on how balanced each ranking is
        # # Can't assume that each alternative is ranked the same number of times
        # proposal_split = np.zeros((m, 2))
        # for ranking in s1:
        #     for cand in ranking:
        #         proposal_split[cand][0] += 1
        # for ranking in s2:
        #     for cand in ranking:
        #         proposal_split[cand][1] += 1
        #
        # proposal_split = np.min(proposal_split, 1)
        # weights = ((gamma ** proposal_split) - 1) / (gamma ** (int(l / 2)) - 1)

        s1 = np.array(s1)
        s2 = np.array(s2)
        profile = np.array(profile)

        weights = []
        for a in range(m):
            # count how many times a appears in both splits
            min_occurrences = min(np.count_nonzero(s1 == a), np.count_nonzero(s2 == a))
            total_occurrences = np.count_nonzero(profile == a)
            total_occurrences += 0.00001
            weights.append(((gamma ** min_occurrences) - 1) / ((gamma ** total_occurrences) - 1))

        dist = kt_distance_between_rankings(ranking1, ranking2, weights=weights)

        all_distances.append(dist)

    return np.mean(all_distances), np.std(all_distances)


def kt_distance_between_many_profiles_with_positional_scoring_rule(profiles, n_splits, score_vector):
    all_dists = []
    for profile in profiles:
        all_splits = make_split_indices(profile, n_splits)
        for split in all_splits:
            s1, s2 = splits_from_split_indices(profile, split_indices=split)
            s1 = pref_voting.profiles.Profile(s1)
            s2 = pref_voting.profiles.Profile(s2)
            ranking1 = vu.positional_scoring_scores(s1, score_vector=score_vector)
            ranking1 = vu.scores_to_tuple_ranking(ranking1)
            ranking2 = vu.positional_scoring_scores(s2, score_vector=score_vector)
            ranking2 = vu.scores_to_tuple_ranking(ranking2)
            dist = kt_distance_between_rankings(ranking1, ranking2)
            all_dists.append(dist)

    mean_kt_dist = np.mean(all_dists)
    return mean_kt_dist


def shuffle_profile(profile, topk, max_size=22000):
    """
    Return a profile where the top k preferences of each voter have been replaced by all possible permutations of
    those rankings.
    :param profile:
    :param topk:
    :param max_size: Arbitrarily chosen to avoid accidentally having a few million voters and taking a long time
    :return:
    """
    assert topk < len(profile[0])

    if topk == 0:
        return profile

    # rankings = profile.rankings
    new_rankings = []
    for ranking in profile:
        top, bottom = ranking[:topk], ranking[topk:]
        perms = itertools.permutations(top)
        for p in perms:
            # new_rankings.append(np.array(p) + bottom)
            new_rankings.append(np.concatenate((np.asarray(p), bottom)))

        if len(new_rankings) > max_size:
            raise ValueError("Given parameters exceeded maximum allowed size for rankings in a profile.")

    # profile = pref_voting.profiles.Profile(rankings=new_rankings)
    return np.asarray(new_rankings)


def evaluate_one_rule(rule, profile, splits, m, k, l, reference_ranking=None, split_type="equal_size", weighted=False):
    """
    Find the average kendall tau distance between the rankings returned by the rule for many random splits.
    If n_splits is an integer, consider that many random splits of the given profile into two profiles.
    :param rule: The rule being considered. The function itself, not the rule name.
    :param profile: list of lists where each list is a ranking over alternatives.
    :param splits: List of pairs; each pair has two index lists corresponding to voters in each split.
    :param reference_ranking: The "ground truth" ranking for the current distribution. Does not always exist.
    :param weighted: Unsupported. Should be False. Will weight splits based on how much each alternatives is represented.
    :return: Mean KT distance between each split. If reference_ranking is not None, also return average distance from
    the outcome on each split to the reference ranking.
    """
    debugging = False
    if debugging:
        print("\n############################")
        print(f"Evaluating {rule.name}")
        print("############################\n")

    all_kt_dists = []
    all_reference_distances = []

    all_s1 = []
    all_s2 = []
    for split in splits:
        s1, s2 = splits_from_split_indices(profile, split)
        all_s1.append(s1)
        all_s2.append(s2)

    in_annealing = False
    kwargs = {"m": m, "k": k}  # pass number of alternatives and length of each preference order
    if rule is vu.annealing_ranking_from_splits:
        in_annealing = True
        kwargs = {"m": m, "k": k, "n_splits": len(splits), "return_vector": True}
        # get mean kt distance between splits
        _, annealed_vector = rule(profile, all_s1, all_s2, **kwargs)
        kwargs = {"m": m, "k": k}
        rule = lambda prf, **kw: vu.scores_to_tuple_ranking(
            vu.positional_scoring_scores(prf, score_vector=annealed_vector, **kw))

    for s1, s2 in zip(all_s1, all_s2):
        if rule is vu.annealing_ranking or rule is vu.annealing_ranking_from_splits:
            kwargs["n_splits"] = len(splits)

        if rule is vu.trimmed_borda_ranking:
            ranking1, ranking2 = vu.compute_trimmed_borda_from_splits(s1, s2, **kwargs)
        else:
            ranking1 = vu.profile_ranking_from_rule(rule, s1, **kwargs)
            ranking2 = vu.profile_ranking_from_rule(rule, s2, **kwargs)

        weights = vu._weight_of_splits(m=m, l=l, s1=s1, s2=s2)

        # find distance between the rankings
        # dist_unweighted = kt_distance_between_rankings(ranking1, ranking2)
        dist = kt_distance_between_rankings(ranking1, ranking2, weights=weights)
        all_kt_dists.append(dist)

        if debugging:
            print(f"Split 1:")
            pprint.pprint(s1)
            print(f"with ranking {ranking1}")

            print(f"Split 2:")
            pprint.pprint(s2)
            print(f"with ranking {ranking2}")

            print(f"Weights are {weights}")
            print()

        if debugging and in_annealing:
            kwargs2 = {"m": m, "k": k}
            ranking1_borda = vu.profile_ranking_from_rule(vu.borda_ranking, s1, **kwargs2)
            ranking2_borda = vu.profile_ranking_from_rule(vu.borda_ranking, s2, **kwargs2)
            borda_dist = kt_distance_between_rankings(ranking1_borda, ranking2_borda, weights=weights)
            if borda_dist < dist:
                print("Borda is better than annealing")

        if reference_ranking is not None:
            # measure distance between rule output on complete profile (no splits) and reference ranking
            # whole_profile = pref_voting.profiles.Profile(profile)
            if rule is vu.trimmed_borda_ranking:
                fake_split = np.asarray([()])
                ref_output, _ = vu.compute_trimmed_borda_from_splits(s1=profile, s2=fake_split, **kwargs)
            else:
                ref_output = vu.profile_ranking_from_rule(rule, profile, **kwargs)
            ref_dist = kt_distance_between_rankings(reference_ranking, ref_output)
            all_reference_distances.append(ref_dist)

    # aggregate all distances together
    kt_mean = np.mean(all_kt_dists)
    kt_std = np.std(all_kt_dists)
    if reference_ranking is not None:
        ref_kt_mean = np.mean(all_reference_distances)
        ref_kt_std = np.std(all_reference_distances)
    else:
        ref_kt_mean = 0
        ref_kt_std = 0

    if debugging:
        print(f"Found mean KT dist =  {kt_mean}")
        print(f"Found KT dist to reference = {ref_kt_mean}")

    if in_annealing:
        return kt_mean, kt_std, ref_kt_mean, ref_kt_std, annealed_vector
    else:
        return kt_mean, kt_std, ref_kt_mean, ref_kt_std


def compare_rules(rules, profiles, n_splits, reference_ranking=None, split_type="equal_size", **kwargs):
    """
    Split profiles into many pairs and find the KT distance between each pair for each rule of interest.
    Also find distance between rule outputs and a provided reference ranking, as applicable.
    :param rules:
    :param profiles:
    :param n_splits:
    :param reference_ranking:
    :param split_type:
    :param kwargs:
    :return:
    """
    if isinstance(profiles, list):
        pass
    elif "n" in kwargs and "dist" in kwargs and "m" in kwargs and isinstance(profiles, int):

        assignments = vu.generate_assignments(n=kwargs["n"], m=kwargs["m"], k=kwargs["k"], l=kwargs["l"])
        profiles = du.generate_profiles(distribution=kwargs["dist"],
                                        profiles_per_distribution=profiles,
                                        num_voters=kwargs["n"],
                                        num_candidates=kwargs["m"],
                                        candidates_per_voter=kwargs["k"],
                                        assignments=assignments,
                                        **kwargs)
    else:
        raise ValueError("Unable to find or generate profiles.")

    if "shuffle_amount" in kwargs:
        k = kwargs["shuffle_amount"]
        profiles = [shuffle_profile(pr, k) for pr in profiles]

    # Generate splits over data so each rule gets the same splits
    splits = [make_split_indices(rankings=profile,
                                 n_splits=n_splits,
                                 split_type=split_type) for profile in profiles]

    results_by_rule = dict()
    annealing_results = []
    for rule in rules:
        print(f"Beginning to evaluate {rule.name}")
        all_dist_means, all_ref_dist_means = [], []
        all_dist_stds, all_ref_dist_stds = [], []

        if rule == vu.choix_pl_ranking and "shuffle_amount" in kwargs and kwargs["shuffle_amount"] >= 3:
            print("Skipping PL MLE because there are too many voters after shuffling.")
            continue

        for p_idx, profile in enumerate(profiles):
            # Find distance between each split for current rule
            # result_values = kt_dist_mean, kt_dist_std, ref_kt_dist_mean, ref_kt_dist_std
            # OR (if using annealing)
            # result_values = kt_dist_mean, kt_dist_std, ref_kt_dist_mean, ref_kt_dist_std, annealing_vector
            result_values = evaluate_one_rule(rule=rule,
                                              profile=profile,
                                              splits=splits[p_idx],
                                              m=kwargs["m"],
                                              k=kwargs["k"],
                                              l=kwargs["l"],
                                              reference_ranking=reference_ranking,
                                              split_type=split_type,
                                              weighted=False)
            all_dist_means.append(result_values[0])
            all_dist_stds.append(result_values[1])
            all_ref_dist_means.append(result_values[2])
            all_ref_dist_stds.append(result_values[3])
            if len(result_values) > 4:
                score_vec = vu.prettify_positional_scores(result_values[4], rounding=3)
                annealing_results.append([rule.name, kwargs["dist"], n_splits, len(profile), kwargs["m"], kwargs["k"], score_vec, result_values[0]])
        results_by_rule[rule.name] = (all_dist_means, all_dist_stds, all_ref_dist_means, all_ref_dist_stds)

    if len(annealing_results) == 0:
        return results_by_rule
    else:
        return results_by_rule, annealing_results


def evaluate_splits_v_ground_truth_all_param_combos(all_n, all_m, k, l, all_dists, all_rules, n_profiles, n_splits,
                                                    parameter_reps,
                                                    split_types=["equal_size"], save_results=True, path="",
                                                    filename="results.csv"):
    columns = ["num_voters", "num_alternatives", "Num Profiles", "profile_set_idx", "Splits Per Profile", "split type",
               "preference distribution", "voting rule",
               "KT Distance Between Splits", "KT Distance Std", "Distance from Central Vote",
               "Central Vote Distance Std"]

    result_rows = []

    profile_set_idx = 0
    annealing_results = []
    anneal_filename = f"anneal_score_vectors.csv"
    anneal_path = os.path.join(path, "annealing_vectors")
    for n, m, pref_dist, split_type in itertools.product(all_n, all_m, all_dists, split_types):
        for pr in range(parameter_reps):
            print(f"Trial {pr} of {parameter_reps} with n={n}, m={m}, pref_dist={pref_dist} and ALL rules.")

            reference_ranking = [(i,) for i in range(m)]
            args = dict()

            results = compare_rules(all_rules,
                                    profiles=n_profiles,
                                    n_splits=n_splits,
                                    reference_ranking=reference_ranking,
                                    split_type=split_type,
                                    n=n,
                                    m=m,
                                    k=k,
                                    l=l,
                                    dist=pref_dist,
                                    **args)
            if isinstance(results, tuple):
                # we are in annealing and have also returned a list of score vectors which should be saved
                # in a separate file
                results, annealed_scores = results
                annealing_results += annealed_scores
                # [rule.name, kwargs["dist"], n_splits, len(profile), kwargs["m"], kwargs["k"], score_vec]

            # Add a result row for each rule evaluated on the profiles
            for rule_name, (dist_means, dist_stds, ref_dist_means, ref_dist_stds) in results.items():
                dist_mean, dist_std = np.mean(dist_means), np.mean(dist_stds)
                ref_dist_mean, ref_dist_stds = np.mean(ref_dist_means), np.mean(ref_dist_stds)

                result_rows.append(
                    [n, m, n_profiles, profile_set_idx, n_splits, split_type, pref_dist, rule_name,
                     round(dist_mean, 5), round(dist_std, 5), round(ref_dist_mean, 5), round(ref_dist_stds, 5)])

            # track which set of profiles we are on
            profile_set_idx += 1

            # Save once in a while so not much gets lost if exiting early
            if save_results:
                df = pd.DataFrame(result_rows, columns=columns)
                # df.sort_values(by="KT Distance Between Splits", ascending=True, inplace=True)

                if not os.path.exists(path):
                    os.makedirs(path)
                df.to_csv(os.path.join(path, filename), index=False)

                if not os.path.exists(anneal_path):
                    os.makedirs(anneal_path)
                anneal_columns = ["Rule Name", "Distribution", "n_splits", "n", "m", "k", "Annealed Scores", "Split Distance"]
                annealing_df = pd.DataFrame(annealing_results, columns=anneal_columns)
                annealing_df.sort_values(by=["Distribution", "Split Distance"], ascending=True, inplace=True)
                annealing_df.to_csv(os.path.join(anneal_path, anneal_filename), index=False)

    df = pd.DataFrame(result_rows, columns=columns)
    # df.sort_values(by="KT Distance Between Splits", ascending=True, inplace=True)

    return df


def compare_basic_ground_truth_vs_split_distance():
    all_dists = ["MALLOWS-0.4", "plackett_luce"]
    all_n = [100]  # Each number of voters/reviewers to experiment over; 100 for paper experiments
    all_m = [100]  # Each number of alternatives/issues to experiment over; 100 for paper experiments
    k = 10  # papers per reviewer (length of each ranking); 10 for paper experiments
    l = k  # reviewers per paper (voters per candidate)
    splits_per_profile = 10  # On each profile, find pairwise distance between this many random splits; 10 for paper
    split_types = ["equal_prob"]    # IJCAI used "equal_size"; now moving to "equal_prob"
    # split_types = ["equal_size"]    # IJCAI used "equal_size"; now moving to "equal_prob"
    n_profiles = 1  # Number of profiles tested during each repetition of parameters; 1 for paper
    parameter_repetitions = 50  # Run this many trials over all combinations of other parameters; 50 for paper
    filename = f"experiment-ground_truth_vs_split_distance-testing-nsplits={splits_per_profile}-trimmed_test.csv"

    all_rules = [
        # vu.annealing_ranking_from_splits,
        # vu.kemeny_gurobi_lazy,
        # vu.choix_pl_ranking,
        # vu.borda_ranking,
        vu.trimmed_borda_ranking,
        # vu.borda_minmax_ranking,
        vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.two_approval_ranking,
        # # vu.copeland_ranking,
        # # vu.dowdall_ranking,
        # # vu.three_approval,
        # # vu.four_approval,
        # # vu.five_approval,
        # # vu.six_approval,
        # # vu.random_ranking,
    ]
    df = evaluate_splits_v_ground_truth_all_param_combos(all_n, all_m, k, l, all_dists, all_rules,
                                                         n_profiles=n_profiles,
                                                         n_splits=splits_per_profile,
                                                         split_types=split_types,
                                                         parameter_reps=parameter_repetitions,
                                                         save_results=True,
                                                         path="results",
                                                         filename=filename)
    return df


def compare_top_shuffling_ground_truth_vs_split_distance():
    all_dists = ["plackett_luce-R", "plackett_luce", "MALLOWS-RELPHI-R", "MALLOWS-RELPHI-0.5"]
    all_dists = ["plackett_luce-R", "plackett_luce"]
    all_dists = ["MALLOWS-RELPHI-R"]
    # all_dists = ["plackett_luce"]
    all_n = [100, 20]  # Each number of voters/reviewers to experiment over
    all_m = [10, 20]  # Each number of alternatives/issues to experiment over
    splits_per_profile = 20  # On each profile, find pairwise distance between this many random splits
    n_profiles = 5  # Number of profiles tested during each repetition of parameters
    parameter_reps = 20  # Run this many trials over all combinations of other parameters

    all_n = [50, 20, 10]  # Each number of voters/reviewers to experiment over
    all_m = [20]  # Each number of alternatives/issues to experiment over
    splits_per_profile = 20  # On each profile, find pairwise distance between this many random splits
    n_profiles = 1  # Number of profiles tested during each repetition of parameters
    parameter_reps = 10  # Run this many trials over all combinations of other parameters
    all_rules = [
        # vu.annealing_ranking,
        vu.kemeny_gurobi,
        vu.choix_pl_ranking,
        # vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.borda_ranking,
        # vu.dowdall_ranking,
        # vu.antiplurality_ranking,
        vu.copeland_ranking,
        # vu.two_approval,
        # vu.three_approval,
        # vu.four_approval,
        # vu.five_approval,
        # vu.six_approval,
        # vu.random_ranking,
    ]
    split_types = ["equal_size"]

    save_results = True
    path = "results"
    filename = "experiment-shuffle_top_preferences-plackett_luce.csv"
    filename = "experiment-shuffle_top_preferences-mallows.csv"

    columns = ["num_voters", "num_alternatives", "Num Profiles", "Shuffle Amount", "Splits Per Profile", "split type",
               "preference distribution", "voting rule",
               "KT Distance Between Splits", "KT Distance Std", "Distance from Central Vote",
               "Central Vote Distance Std"]
    result_rows = []

    shuffle_amounts = [4, 3, 2, 0]
    shuffle_amounts = [0]
    for sh in shuffle_amounts:
        args = dict()
        args["shuffle_amount"] = sh
        for n, m, pref_dist, split_type in itertools.product(all_n, all_m, all_dists, split_types):

            # if sh >= 3 and rule == vu.choix_pl_ranking:
            #     continue
            for pr in range(parameter_reps):
                print(
                    f"Trial {pr} of {parameter_reps} with n={n}, m={m}, pref_dist={pref_dist}, shuffles={sh}")

                reference_ranking = [(i,) for i in range(m)]

                # if pref_dist == "plackett_luce":
                #     alpha = 0.5
                #     alphas = [np.exp(alpha * i) for i in range(m, 0, -1)]
                #     args["alphas"] = alphas

                # profiles = [du.generate_profile(distribution=pref_dist,
                #                                 num_voters=n,
                #                                 num_candidates=m,
                #                                 **args) for _ in range(n_profiles)]

                results_by_rule = compare_rules(all_rules,
                                                profiles=n_profiles,
                                                n_splits=splits_per_profile,
                                                reference_ranking=reference_ranking,
                                                split_type=split_type,
                                                n=n,
                                                m=m,
                                                dist=pref_dist,
                                                **args)

                # Add a result row for each rule evaluated on the profiles
                for rule_name, (dist_means, dist_stds, ref_dist_means, ref_dist_stds) in results_by_rule.items():
                    dist_mean, dist_std = np.mean(dist_means), np.mean(dist_stds)
                    ref_dist_mean, ref_dist_stds = np.mean(ref_dist_means), np.mean(ref_dist_stds)

                    result_rows.append(
                        [n, m, n_profiles, sh, splits_per_profile, split_type, pref_dist, rule_name,
                         round(dist_mean, 3), round(dist_std, 3), round(ref_dist_mean, 3), round(ref_dist_stds, 3)])

            # Save once in a while so not much gets lost if exiting early
            if save_results:
                df = pd.DataFrame(result_rows, columns=columns)
                # df.sort_values(by="KT Distance Between Splits", ascending=True, inplace=True)

                if not os.path.exists(path):
                    os.makedirs(path)
                df.to_csv(os.path.join(path, filename), index=False)

    df = pd.DataFrame(result_rows, columns=columns)
    # df.sort_values(by="KT Distance Between Splits", ascending=True, inplace=True)

    return df


def rule_picking_rule(profile, possible_rules, n_splits, split_type="equal_size", count_ties=False):
    """
    Given a preference order and a list of possible rules, find the rule which minimizes KT distance over the specified
    number of splits on this profile.
    :param profile:
    :param possible_rules:
    :param n_splits:
    :param split_type:
    :return: The function corresponding to the rule with the minimal KT distance averaged over all splits on the
    given profile.
    """
    rule_dists = {rule: [] for rule in possible_rules}  # track kt dist of each rule output across each split

    kwargs = {"m": len(profile[0]),
              "k": len(profile[0])}  # pass number of alternatives and length of each preference order

    for rule in possible_rules:
        splits = make_split_indices(profile, n_splits, split_type=split_type)
        for split in splits:
            # print(f"Evaluating split {split_idx} of {rule}")
            s1, s2 = splits_from_split_indices(profile, split_indices=split)
            # s1, s2 = splits_from_split_indices(profile, split)
            # s1, s2 = split_data(rankings=profile, n=len(profile), m=m, split_type=split_type)

            # find rankings
            # s1 = pref_voting.profiles.Profile(s1)
            # s2 = pref_voting.profiles.Profile(s2)
            ranking1 = vu.profile_ranking_from_rule(rule, s1, **kwargs)
            ranking2 = vu.profile_ranking_from_rule(rule, s2, **kwargs)

            # find distance between the rankings
            dist = kt_distance_between_rankings(ranking1, ranking2)
            rule_dists[rule].append(dist)

    if count_ties:
        mean_dists = {rule: np.mean(rule_dists[rule]) for rule in possible_rules}
        best_dist = min(mean_dists.values())
        epsilon = 0.00001
        best_rules = [rule for rule, dist in mean_dists.items() if dist - best_dist < epsilon]

        best_rules = sorted(best_rules)[0]
        num_best_rules = len(best_rules)
        return best_rules, num_best_rules
    else:
        mean_dists = {rule: np.mean(rule_dists[rule]) for rule in possible_rules}
        best_rule = min(mean_dists, key=mean_dists.get)
        return best_rule

    return best_rule


def generate_binary_matrix(n, m, assignments_per_reviewer, reviewers_per_assignment):
    # Create an empty matrix

    assert n == m
    assert assignments_per_reviewer == reviewers_per_assignment
    assert n == assignments_per_reviewer ** 2
    matrix = np.zeros((n, m), dtype=int)

    ones_per_col = reviewers_per_assignment
    ones_per_row = assignments_per_reviewer

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


if __name__ == "__main__":

    df = compare_basic_ground_truth_vs_split_distance()
