import collections

import pandas as pd

import rule_comparison
from utils import data_utils as du
from utils import voting_utils as vu
import rule_comparison as rc
import numpy as np
from preflib import preflib
import itertools
from scipy.stats import sem


def evaluate_olympics_data():
    olympics_event_detail, olympics_weak_rankings_dict = du.load_olympic_rankings(include_noncompeting_countries=False,
                                                                                  include_nonmedaling_competitors=False,
                                                                                  maxsize=None)

    results = []

    # consider each event separately:
    for game, event_rankings in olympics_weak_rankings_dict.items():

        print(f"Beginning results for: {game}")
        print("Game, n_splits, n_events, rule_name, distance, distance_std")

        # evaluate one set of weak rankings as per "usual"
        distance_results, all_rule_distances = evaluate_rankings_olympics(event_rankings, include_annealing=True)

        for result in distance_results:
            results.append([game] + result)

        # print(results)
        print("\n")

    # [n_splits, num_countries, len(rankings),  rule.name, dist, std]
    results = pd.DataFrame(data=results,
                           columns=["Game", "n_splits", "n_countries", "n_events", "rule_name", "distance",
                                    "distance_std", "winning_ranking", "annealed_scores"])

    results.sort_values(by=["Game", "distance"], ascending=True, inplace=True)

    results.to_csv("results/olympic_data-neurips.csv", index=False)


def evaluate_rankings_olympics(rankings, include_annealing=False, include_kemeny=False):
    """
    Using the set list of rules and other parameters, return the KT distance of each rule on the given rankings.
    :param rankings: A weak ranking consisting of list of tuples.
    :return:
    """

    all_rule_distances = dict()
    distance_results = []

    n_splits = 10
    split_type = "equal_prob"

    num_countries = 0
    for ranking in rankings:
        for tup in ranking:
            for num in tup:
                if num > num_countries:
                    num_countries = num
    num_countries += 1

    n = len(rankings)
    m = num_countries

    all_rules = []
    if include_annealing:
        all_rules.append(vu.annealing_ranking_from_splits)
    all_rules += [
        vu.olympic_medal_count_ranking,
        vu.olympic_gold_count_ranking,
        vu.borda_ranking,
        vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.two_approval_ranking,
    ]
    all_rule_score_vectors = [
        vu.f1_1991_ranking_vector(3),
        vu.f1_2003_ranking_vector(3),
        vu.f1_2010_ranking_vector(3),
        vu.borda_ranking_vector(3),
        vu.plurality_ranking_vector(3),
        vu.plurality_veto_ranking_vector(3),
        vu.antiplurality_ranking_vector(3),
        vu.two_approval_ranking_vector(3),
        vu.olympic_medal_count_ranking_vector(rankings, 3),
        vu.olympic_gold_count_ranking_vector(3)
    ]
    all_rule_score_vectors = [vu.prettify_positional_scores(sv) for sv in all_rule_score_vectors]
    all_rule_score_vectors.sort()
    all_rule_score_vectors = list(k for k, _ in itertools.groupby(all_rule_score_vectors))
    # all_rule_score_vectors = list(set(all_rule_score_vectors))

    if include_kemeny:
        # Note: Kemeny and PL MLE will not work with Olympic data. They require every pref order include an equal
        # number of candidates. Only the case here if we include countries that DO NOT compete which is entirely
        # infeasible computationally
        all_rules.append(vu.kemeny_gurobi_lazy)

    all_s1 = []
    all_s2 = []
    # all_splits = []
    for _ in range(n_splits):
        s1, s2 = rc.split_data(rankings, n=n, m=m, split_type=split_type)
        # all_splits.append([s1, s2])
        all_s1.append(s1)
        all_s2.append(s2)

    # for rule in all_rules:
    #     dist, std = kt_distance_one_profile_one_rule(profile=rankings, all_s1=all_s1, all_s2=all_s2, rule=rule,
    #                                                  m=num_countries,
    #                                                  use_mean_score_on_ties=False,  # rank each candidate individually
    #                                                  normalize=False
    #                                                  )
    #
    #     all_rule_distances[rule.name] = (dist, std)
    #     distance_results.append([n_splits, num_countries, len(rankings), rule.name, round(dist, 3), round(std, 3)])
    #
    #     print(f"{round(dist, 5)} ± {round(std, 6)} is KT Distance of {rule.name}.")

    for rule in all_rules:
        dist, std, winner, annealed_scores = kt_distance_one_profile_one_rule(profile=rankings,
                                                                              all_s1=all_s1,
                                                                              all_s2=all_s2,
                                                                              rule=rule,
                                                                              m=num_countries,
                                                                              use_mean_score_on_ties=False,
                                                                              # rank each candidate individually
                                                                              return_winner=True,
                                                                              normalize=False,
                                                                              all_annealing_states=all_rule_score_vectors
                                                                              )

        all_rule_distances[rule.name] = (dist, std)
        distance_results.append(
            [n_splits, num_countries, len(rankings), rule.name, round(dist, 5), round(std, 5), winner, annealed_scores])

        print(f"{round(dist, 5)} ± {round(std, 6)} is KT Distance of {rule.name}.")

    return distance_results, all_rule_distances


def evaluate_rankings_alma(rankings, num_alternatives, normalize, include_rule=None, include_annealing=False,
                           include_kemeny=False):
    """
    Same function as others but set up with defaults suitable for ALMA data/matching what was done in paper.
    Evaluate the given weak rankings over many rules. Return the KT distance of each rule on the profile.
    :param rankings: One profile containing weak rankings over alternatives. Assume, for now, that all rankings are
    complete. Complete rankings by adding missing alternatives to tied positions at the bottom of each preference order.
    :param num_alternatives:
    :param include_rule: None or a function that is itself suitable as a voting rule.
    :param include_annealing:
    :param include_kemeny:
    :return:
    """

    n_splits = 10
    split_type = "equal_prob"

    m = 0
    for ranking in rankings:
        for tup in ranking:
            for num in tup:
                if num > m:
                    m = num
    m += 1
    n = len(rankings)
    k = sum([len(tied_cands) for tied_cands in rankings[0]])  # assume everyone ranks equal number (?)

    all_rules = [
        vu.trimmed_borda_ranking,
        vu.borda_minmax_ranking,
        # vu.choix_pl_ranking,
        vu.borda_ranking,
        vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.two_approval_ranking,
        vu.three_approval_ranking,
        vu.seven_approval_ranking,
        vu.eight_approval_ranking,
        # vu.nine_approval_ranking,
    ]
    all_rule_score_vectors = [
        vu.borda_ranking_vector(k),
        # vu.plurality_ranking_vector(k),
        # vu.plurality_veto_ranking_vector(k),
        vu.antiplurality_ranking_vector(k),
        # vu.two_approval_ranking_vector(k),
    ]
    all_rule_score_vectors = [vu.prettify_positional_scores(sv) for sv in all_rule_score_vectors]
    if include_rule:
        all_rules.append(include_rule)
    if include_annealing:
        all_rules.append(vu.annealing_ranking_from_splits)
    if include_kemeny:
        print(f"Using Kemeny. MAKE SURE TO SET TIME_OUT IN RULE. UNSET WHEN DONE.")
        all_rules.append(vu.kemeny_gurobi_lazy)

    all_rule_distances = dict()
    distance_results = []

    all_s1 = []
    all_s2 = []
    # all_splits = []
    for _ in range(n_splits):
        s1, s2 = rc.split_data(rankings, n=n, m=m, split_type=split_type)
        # all_splits.append([s1, s2])
        all_s1.append(s1)
        all_s2.append(s2)

    for rule in all_rules:
        dist, std, winner, annealed_scores = kt_distance_one_profile_one_rule(profile=rankings,
                                                                              all_s1=all_s1,
                                                                              all_s2=all_s2,
                                                                              rule=rule,
                                                                              m=num_alternatives,
                                                                              return_winner=True,
                                                                              k=k,
                                                                              normalize=normalize,
                                                                              all_annealing_states=all_rule_score_vectors,
                                                                              )

        all_rule_distances[rule.name] = (dist, std)
        distance_results.append(
            [n_splits, num_alternatives, len(rankings), rule.name, dist, std, winner,
             annealed_scores])

        print(f"{round(dist, 5)} ± {round(std, 6)} is KT Distance of {rule.name}.")

    return distance_results, all_rule_distances


def evaluate_rankings(rankings, num_alternatives, normalize, include_rule=None, include_annealing=False,
                      include_kemeny=False):
    """
    Evaluate the given weak rankings over many rules. Return the KT distance of each rule on the profile.
    :param rankings: One profile containing weak rankings over alternatives. Assume, for now, that all rankings are
    complete. Complete rankings by adding missing alternatives to tied positions at the bottom of each preference order.
    :param num_alternatives:
    :param include_rule: None or a function that is itself suitable as a voting rule.
    :param include_annealing:
    :param include_kemeny:
    :return:
    """

    n_splits = 10
    split_type = "equal_prob"

    m = 0
    for ranking in rankings:
        for tup in ranking:
            for num in tup:
                if num > m:
                    m = num
    m += 1
    n = len(rankings)
    k = sum([len(tied_cands) for tied_cands in rankings[0]])  # assume everyone ranks equal number (?)

    all_rules = [
        vu.trimmed_borda_ranking,
        vu.borda_minmax_ranking,
        vu.plurality_ranking,
        vu.f1_1991_ranking,
        vu.f1_2003_ranking,
        vu.f1_2010_ranking,
        vu.borda_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.two_approval_ranking,
        # vu.kemeny_gurobi_lazy,
        # vu.choix_pl_ranking,
        # vu.copeland_ranking,
        # vu.dowdall_ranking,
        # vu.three_approval,
        # vu.four_approval,
        # vu.five_approval,
        # vu.six_approval,
        # vu.random_ranking,
    ]
    all_rule_score_vectors = [
        vu.f1_1991_ranking_vector(m),
        vu.f1_2003_ranking_vector(m),
        vu.f1_2010_ranking_vector(m),
        vu.borda_ranking_vector(m),
        vu.plurality_ranking_vector(m),
        vu.plurality_veto_ranking_vector(m),
        vu.antiplurality_ranking_vector(m),
        vu.two_approval_ranking_vector(m),
    ]
    all_rule_score_vectors = [vu.prettify_positional_scores(sv) for sv in all_rule_score_vectors]
    if include_rule:
        all_rules.append(include_rule)
    if include_annealing:
        all_rules.append(vu.annealing_ranking_from_splits)
    if include_kemeny:
        all_rules.append(vu.kemeny_gurobi_lazy)

    all_rule_distances = dict()
    distance_results = []

    all_s1 = []
    all_s2 = []
    # all_splits = []
    for _ in range(n_splits):
        s1, s2 = rc.split_data(rankings, n=n, m=m, split_type=split_type)
        # all_splits.append([s1, s2])
        all_s1.append(s1)
        all_s2.append(s2)

    for rule in all_rules:
        dist, std, winner, annealed_scores = kt_distance_one_profile_one_rule(profile=rankings,
                                                                              all_s1=all_s1,
                                                                              all_s2=all_s2,
                                                                              rule=rule,
                                                                              m=num_alternatives,
                                                                              return_winner=True,
                                                                              k=k,
                                                                              normalize=normalize,
                                                                              all_annealing_states=all_rule_score_vectors,
                                                                              time_out=10)

        all_rule_distances[rule.name] = (dist, std)
        distance_results.append(
            [n_splits, num_alternatives, len(rankings), rule.name, dist, std, winner,
             annealed_scores])

        print(f"{round(dist, 5)} ± {round(std, 6)} is KT Distance of {rule.name}.")

    return distance_results, all_rule_distances


def kt_distance_one_profile_one_rule(profile, all_s1, all_s2, rule, m, return_winner=False, **kwargs):
    """
    Measure the KT distance between splits on one given profile using a single rule.
    Assume that the profile may contain weak rankings and that it provides complete information
    :param profile:
    :param all_s1:
    :param all_s2:
    :param rule:
    :param m:
    :param return_winner: If True, return the ranking of the rule on the profile
    :return:
    """
    all_distances = []
    n = len(profile)
    if "k" in kwargs:
        k = kwargs["k"]
    else:
        k = len(profile[0])  # number of candidates ranked by each voter (potential positions per event: 3 medals)

    if "normalize" in kwargs:
        normalize = kwargs["normalize"]
    else:
        normalize = True

    annealed_vector = []

    if rule is vu.annealing_ranking_from_splits:
        in_annealing = True
        if "use_mean_score_on_ties" in kwargs:
            use_mean_score_on_ties = kwargs["use_mean_score_on_ties"]
        else:
            use_mean_score_on_ties = True
        if "all_annealing_states" in kwargs:
            all_annealing_states = kwargs["all_annealing_states"]
        else:
            all_annealing_states = [
                [(m - i - 1) / m for i in range(m)]  # If not specified, start only from Borda
            ]
        kwargs = {"m": m, "k": k,
                  "n_splits": len(all_s1),
                  "return_vector": True,
                  "use_mean_score_on_ties": use_mean_score_on_ties,
                  "normalize": normalize,
                  "all_annealing_states": all_annealing_states
                  }
        # get mean kt distance between splits
        _, annealed_vector = rule(profile, all_s1, all_s2, **kwargs)
        kwargs = {"m": m, "k": k}
        rule = lambda prf, **kw: vu.scores_to_tuple_ranking(
            vu.positional_scoring_scores(prf,
                                         score_vector=annealed_vector,
                                         use_mean_score_on_ties=use_mean_score_on_ties,
                                         **kw))

    for s1, s2 in zip(all_s1, all_s2):

        kwargs = {"m": m, "k": k,
                  "normalize": normalize}

        if rule is vu.trimmed_borda_ranking:
            ranking1, ranking2 = vu.compute_trimmed_borda_from_splits(s1, s2, **kwargs)
        else:
            ranking1 = vu.profile_ranking_from_rule(rule, s1, **kwargs)
            ranking2 = vu.profile_ranking_from_rule(rule, s2, **kwargs)

        if normalize:
            gamma = 2
            flat_s1 = [alternative for order in s1 for tied_alternatives in order for alternative in tied_alternatives]
            flat_s2 = [alternative for order in s2 for tied_alternatives in order for alternative in tied_alternatives]
            flat_profile = [alternative for order in profile for tied_alternatives in order for alternative in
                            tied_alternatives]

            s1 = np.array(flat_s1)
            s2 = np.array(flat_s2)
            flat_profile = np.array(flat_profile)

            weights = []
            for a in range(m):
                # count how many times each alternative a appears in both splits
                min_occurrences = min(np.count_nonzero(s1 == a), np.count_nonzero(s2 == a))
                total_occurrences = np.count_nonzero(flat_profile == a)
                total_occurrences += 0.00001

                if min_occurrences > 1000:
                    # Slightly approximate to avoid overflow issues
                    exponent = int(min_occurrences - total_occurrences / 2)
                    weights.append(gamma ** exponent)
                else:
                    weights.append(((gamma ** min_occurrences) - 1) / ((gamma ** (total_occurrences / 2)) - 1))
        else:
            weights = [1] * m

        use_jaccard = False
        if use_jaccard:
            print("USING JACCARD INDEX. BEWARE!")
            dist = rc.jaccard_distance_between_rankings(ranking1, ranking2, weights=weights)
        else:
            dist = rc.kt_distance_between_rankings(ranking1, ranking2, weights=weights)

        # print(f"In rule={rule}; dist={dist}")

        all_distances.append(dist)

    if len(annealed_vector) > 0:
        print(f"Before prettification, annealing scores are: {annealed_vector}")
        annealed_vector = vu.prettify_positional_scores(annealed_vector)
        print(f"After prettification, annealing scores are: {annealed_vector}")

    if return_winner:
        winning_ranking = vu.profile_ranking_from_rule(rule, profile, **kwargs)
        # return np.mean(all_distances), np.std(all_distances), winning_ranking, annealed_vector
        print("returning sem")
        return np.mean(all_distances), sem(all_distances), winning_ranking, annealed_vector
    else:
        return np.mean(all_distances), np.std(all_distances), annealed_vector


def evaluate_preflib_data():
    cols = ["Dataset", "n_splits", "n_alternatives", "n_voters", "rule_name", "distance", "distance_std",
            "winning_ranking", "annealed_scores"]
    preflib_results = []

    out_path = "preflib/analysis_results-neurips.csv"

    # #########################
    # Formula 1 Racing
    #########################
    f1_profiles = preflib.load_formula_1_elections()
    for (data_name, profile, rule, m) in f1_profiles:

        weak_ranking = du.weak_ranking_from_strict_ranking(profile, m)
        row_results, dict_results = evaluate_rankings(rankings=weak_ranking,
                                                      num_alternatives=m,
                                                      normalize=False,
                                                      # include_rule=rule,
                                                      include_rule=None,
                                                      include_annealing=True
                                                      )
        for row in row_results:
            preflib_results.append([data_name] + row)

        # If including a rule, check how often other rules match it

    df = pd.DataFrame(columns=cols, data=preflib_results)
    df.to_csv(out_path, index=False)

    # #########################
    # # UK Labour Election Data
    # #########################
    # print("Starting UK Labour")
    # uk_labour_profiles = preflib.load_uk_labour_party_leadership_election()
    # for (data_name, profile, rule, m) in uk_labour_profiles:
    #
    #     weak_ranking = du.weak_ranking_from_strict_ranking(profile, m)
    #     row_results, dict_results = evaluate_rankings(rankings=weak_ranking,
    #                                                   num_alternatives=m,
    #                                                   include_rule=rule,
    #                                                   include_annealing=True)
    #     for row in row_results:
    #         preflib_results.append([data_name] + row)
    #
    # df = pd.DataFrame(columns=cols, data=preflib_results)
    # df.to_csv(out_path, index=False)

    #########################
    # IRV City Election Data
    #########################
    print("Starting City IRV Election Data")
    # city_election_profiles = preflib.load_city_election_data(max_count=100, max_n_voters=16000)
    # city_election_profiles = preflib.load_city_election_data(max_n_voters=16000)
    city_election_profiles = preflib.load_city_election_data()
    print(f"Loaded {len(city_election_profiles)} city elections.")
    for (data_name, profile, rule, m) in city_election_profiles:
        print(f"Starting to measure: {data_name} with {len(profile)} rankings and {m} candidates.")

        # weak_ranking = du.weak_ranking_from_strict_ranking(profile, m)
        row_results, dict_results = evaluate_rankings(rankings=profile,
                                                      num_alternatives=m,
                                                      normalize=False,
                                                      include_rule=rule,
                                                      include_annealing=False)
        for row in row_results:
            preflib_results.append([data_name] + row)

    df = pd.DataFrame(columns=cols, data=preflib_results)
    df.to_csv(out_path, index=False)

    return df


def evaluate_alma_data(file_dir="alma_data", file_name="alma_output.csv"):
    # Load ALMA data
    df = pd.read_csv(f"{file_dir}/{file_name}")
    out_path = f"{file_dir}/results-{file_name}"

    # Format into rankings and profiles
    profile = []
    if file_name == "alma_output.csv":
        # cols = ["Owner_anon", "Assignment_anon", "Rank: final individual"]
        # rename alternatives so they are ints starting from zero
        df['assignment'] = pd.Categorical(df['Assignment_anon']).codes

        for reviewer in df["Owner_anon"].unique():
            ranked_projects = df[df["Owner_anon"] == reviewer]
            ranked_projects = ranked_projects.sort_values("Rank: final individual")

            profile.append([(alt,) for alt in ranked_projects["assignment"]])

    elif file_name == "alma_data_cycle10.csv":
        # cols = ["Reviewer id", "Submission id", "rank", "rating"]
        # rename alternatives so they are ints starting from zero
        df['assignment'] = pd.Categorical(df['Submission id']).codes

        for reviewer in df["Reviewer id"].unique():
            ranked_projects = df[df["Reviewer id"] == reviewer]
            ranked_projects = ranked_projects.sort_values("rank")

            profile.append([(alt,) for alt in ranked_projects["assignment"]])
    else:
        raise ValueError(f"Unexpected path. Should be one of hardcoded values. Got: {file_name}")

    print(f"# rankings {len(profile)}")

    # Run ABA
    m = max(df["assignment"]) + 1

    print(f"# alternatives: {m}")

    rows_to_save = []
    row_results, dict_results = evaluate_rankings_alma(rankings=profile,
                                                       num_alternatives=m,
                                                       normalize=True,
                                                       include_rule=None,
                                                       include_annealing=True,
                                                       include_kemeny=True
                                                       )

    for row in row_results:
        rows_to_save.append(["ALMA"] + row[:-2])
        # preflib_results.append([data_name] + row)

    cols = ["Dataset", "n_splits", "n_alternatives", "n_voters", "rule_name", "distance", "distance_sem",
            # "winning_ranking", "annealed_scores"
            ]
    df = pd.DataFrame(columns=cols, data=rows_to_save)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # evaluate_olympics_data()
    # evaluate_preflib_data()
    evaluate_alma_data(
        file_dir="alma_data",
        file_name="alma_output.csv",
        # file_name="alma_data_cycle10.csv",
    )
    evaluate_alma_data(
        file_dir="alma_data",
        # file_name="alma_output.csv",
        file_name="alma_data_cycle10.csv",
    )
