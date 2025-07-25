import collections
import itertools
from collections import Counter
import random

import pref_voting
import rule_comparison
from utils.decorator import method_name
import rule_comparison as rc
from utils import voting_utils as vu
from utils import data_utils as du
import pandas as pd
import numpy as np


def evaluate_axioms(axioms, profile_set, possible_rules, **kwargs):
    axiom_results = {
        # map of axiom name mapped to dict of total number of possible violations and total number of actual violations
        ax.name: {"possible_violations": 0,
                  "total_violations": 0,
                  "nonviolating_rule_names": 0}
        for ax in axioms
    }

    for ax in axioms:
        print(f"Starting a round of: {ax.name}")

        # Special case: Combined two axioms for large reduction in time
        if ax.name == "Consistency":
            if "Consistency" in axiom_results:
                del axiom_results["Consistency"]

            pv_strong, tv_strong, nvr_strong, pv_weak, tv_weak, nvr_weak = ax(profile_set, possible_rules, **kwargs)
            # possible_violations, total_violations, nonviolating_rule_names = ax(profile_set, possible_rules, **kwargs)
            axiom_results["Strong Consistency"] = {"possible_violations": 0,
                                                   "total_violations": 0,
                                                   "nonviolating_rule_names": 0}
            axiom_results["Weak Consistency"] = {"possible_violations": 0,
                                                 "total_violations": 0,
                                                 "nonviolating_rule_names": 0}
            axiom_results["Strong Consistency"]["possible_violations"] = pv_strong
            axiom_results["Strong Consistency"]["total_violations"] = tv_strong
            axiom_results["Strong Consistency"]["nonviolating_rule_names"] = nvr_strong

            axiom_results["Weak Consistency"]["possible_violations"] = pv_weak
            axiom_results["Weak Consistency"]["total_violations"] = tv_weak
            axiom_results["Weak Consistency"]["nonviolating_rule_names"] = nvr_weak

            continue

        # All axioms other than consistency
        possible_violations, total_violations, nonviolating_rule_names = ax(profile_set, possible_rules, **kwargs)
        axiom_results[ax.name]["possible_violations"] = possible_violations
        axiom_results[ax.name]["total_violations"] = total_violations
        axiom_results[ax.name]["nonviolating_rule_names"] = nonviolating_rule_names

    return axiom_results


@method_name("Strong Consistency")
def strong_consistency(profile_set, possible_rules, **kwargs):
    possible_violations, total_violations = 0, 0
    nonviolating_rule_names = collections.Counter()

    for P in profile_set:
        all_splits = rc.make_split_indices(P, kwargs["n_splits"], split_type="equal_size")
        for split in all_splits:
            s1, s2 = rc.splits_from_split_indices(P, split_indices=split)
            z1 = rc.rule_picking_rule(s1, possible_rules, kwargs["n_splits"])
            z2 = rc.rule_picking_rule(s2, possible_rules, kwargs["n_splits"])

            if z1 != z2:
                # both rules are not the same so violating the axiom is not possible with this profile
                continue

            # profile splits are such that satisfying/violating the axiom is possible

            z_aggregate = rule_comparison.rule_picking_rule(P, possible_rules, kwargs["n_splits"])

            possible_violations += 1
            if z1 != z_aggregate:
                total_violations += 1
            else:
                # unclear how many times to add the rule since they are all the same here
                nonviolating_rule_names[z1.name] += 1
                # nonviolating_rule_names[z2.name] += 1
                # nonviolating_rule_names[z_aggregate.name] += 1

    return possible_violations, total_violations, nonviolating_rule_names


@method_name("Union Consistency")
def weak_consistency(profile_set, possible_rules, **kwargs):
    possible_violations, total_violations = 0, 0
    nonviolating_rule_names = collections.Counter()

    for P in profile_set:
        all_splits = rc.make_split_indices(P, kwargs["n_splits"], split_type="equal_size")
        for split in all_splits:
            s1, s2 = rc.splits_from_split_indices(P, split_indices=split)
            z1 = rc.rule_picking_rule(s1, possible_rules, kwargs["n_splits"])
            z2 = rc.rule_picking_rule(s2, possible_rules, kwargs["n_splits"])

            if z1 != z2:
                # both rules are not the same so violating the axiom is not possible with this profile
                continue

            s1 = pref_voting.profiles.Profile(s1)
            s2 = pref_voting.profiles.Profile(s2)
            if z1(s1) != z2(s2):
                # the rule returned a different ranking on each split
                continue

            # profile splits are such that satisfying/violating the axiom is possible

            z_aggregate = rc.rule_picking_rule(P, possible_rules, kwargs["n_splits"])

            possible_violations += 1
            if z1 != z_aggregate:
                # The rule with lowest KT distance on both splits is different than rule with lowest KT distance on
                # the whole profile
                total_violations += 1
            else:
                # unclear how many times to add the rule since they are all the same here
                nonviolating_rule_names[z1.name] += 1
                # nonviolating_rule_names[z2.name] += 1
                # nonviolating_rule_names[z_aggregate.name] += 1

    return possible_violations, total_violations, nonviolating_rule_names


@method_name("Consistency")
def consistency(profile_set, possible_rules, **kwargs):
    # Implements both consistency axioms
    strong_possible_violations, strong_total_violations = 0, 0
    weak_possible_violations, weak_total_violations = 0, 0
    strong_nonviolating_rule_names = collections.Counter()
    weak_nonviolating_rule_names = collections.Counter()

    for P in profile_set:
        splits_to_evaluate = 1  # Faster for testing, should maybe do all n_splits for publishing
        # splits_to_evaluate = kwargs["n_splits"]
        all_splits = rc.make_split_indices(P, splits_to_evaluate, split_type="equal_size")
        for split in all_splits:
            s1, s2 = rc.splits_from_split_indices(P, split_indices=split)
            z1 = rc.rule_picking_rule(s1, possible_rules, kwargs["n_splits"])
            z2 = rc.rule_picking_rule(s2, possible_rules, kwargs["n_splits"])

            strong_violated = False
            weak_violated = False

            rules_are_same = True
            if z1 != z2:
                # Required for Strong and Weak Consistency
                # both rules are not the same so violating the axiom is not possible with this profile

                rules_are_same = False

            # s1 = pref_voting.profiles.Profile(s1)
            # s2 = pref_voting.profiles.Profile(s2)
            rankings_are_same = True
            if z1(s1, **kwargs) != z2(s2, **kwargs):
                # Required for Weak Consistency only, not Strong Consistency
                # the rule returned a different ranking on each split
                # weak_possible = False
                # strong_violated = True
                # weak_violated = True

                rankings_are_same = False

            strong_possible = rules_are_same
            weak_possible = rules_are_same and rankings_are_same

            if not (weak_possible or strong_possible):
                continue

            # profile splits are such that satisfying/violating the axiom is possible

            z_aggregate = rc.rule_picking_rule(P, possible_rules, kwargs["n_splits"])

            if strong_possible:
                strong_possible_violations += 1
            if weak_possible:
                weak_possible_violations += 1
            if z1 != z_aggregate:
                # The rule with lowest KT distance on both splits is different than rule with lowest KT distance on
                # the whole profile
                if strong_possible:
                    strong_total_violations += 1
                if weak_possible:
                    weak_total_violations += 1
            else:
                # All rules involved (z1, z2, z_aggregate) are the same if we reach here so just add them once
                if strong_possible:
                    strong_nonviolating_rule_names[z1.name] += 1
                if weak_possible:
                    weak_nonviolating_rule_names[z1.name] += 1

    return strong_possible_violations, strong_total_violations, strong_nonviolating_rule_names, weak_possible_violations, weak_total_violations, weak_nonviolating_rule_names


@method_name("Reversal Symmetry")
def reversal_symmetry(profile_set, possible_rules, **kwargs):
    possible_violations, total_violations = len(profile_set), 0
    nonviolating_rule_names = collections.Counter()

    for profile in profile_set:
        Z = rc.rule_picking_rule(profile, possible_rules, kwargs["n_splits"])

        # s = pref_voting.profiles.Profile(profile)
        # Find ranking of rule which minimizes KT dist over many splits
        r1 = Z(profile, **kwargs)

        # Find reversal of ranking from same rule using reversed profiles

        # reverse preference orders
        rev_profile = profile[:, ::-1]
        # s2 = pref_voting.profiles.Profile(rev_profile)

        if Z.reversible:
            r2 = Z(rev_profile, reverse_vector=True, **kwargs)
        else:
            err = "Should not pass rules that are not reversible when testing reversal symmetry."
            err += f"Given: {Z.name}"
            raise ValueError(err)

        # r2 = Z(s2)

        # r2 = list(reversed(r2))

        # satisfied = r1 == r2
        if r1 != r2:
            total_violations += 1
        else:
            nonviolating_rule_names[Z.name] += 1

    return possible_violations, total_violations, nonviolating_rule_names


@method_name("Monotonicity")
def monotonicity(profile_set, possible_rules, **kwargs):
    possible_violations, total_violations = len(profile_set), 0
    nonviolating_rule_names = collections.Counter()

    for profile in profile_set:
        Z = rc.rule_picking_rule(profile, possible_rules, kwargs["n_splits"])
        # s = pref_voting.profiles.Profile(profile)
        r1 = Z(profile, **kwargs)
        # find top alternatives a of F(P)
        top_alternative = r1[0]

        # alter some profiles (decide which as parameter in kwargs?)
        if "shuffle_fraction" in kwargs:
            shuffle_fraction = kwargs["shuffle_fraction"]
        else:
            # totally arbitrarily chosen default shuffle value
            shuffle_fraction = random.uniform(0.2, 0.8)

        profile2 = du.improve_alternatives(alternatives=top_alternative, profile=profile, probability=shuffle_fraction)
        # s2 = pref_voting.profiles.Profile(profile2)
        # find top alternatives of F(P')
        r2 = Z(profile2, **kwargs)
        top_alternative2 = r2[0]
        if top_alternative != top_alternative2:
            total_violations += 1
        else:
            # track which rules have minimal KT distance and do not violate the axioms
            nonviolating_rule_names[Z.name] += 1

    return possible_violations, total_violations, nonviolating_rule_names


@method_name("Homogeneity")
def homogeneity(profile_set, possible_rules, **kwargs):
    possible_violations, total_violations = len(profile_set), 0
    nonviolating_rule_names = collections.Counter()

    idx = 0
    for profile in profile_set:
        print(f"On profile {idx}")
        idx += 1
        # Find a rule-picking-rule for the profile
        Z1 = rc.rule_picking_rule(profile, possible_rules, kwargs["n_splits"])

        # Replace each voter with several copies of itself:
        k = 5  # arbitrarily chosen multiplier
        expanded_profile = du.multiply_profile(profile, k)

        # Find the rule-picking rule again, for the expanded profile
        Z2 = rc.rule_picking_rule(expanded_profile, possible_rules, kwargs["n_splits"])

        if Z1 != Z2:
            total_violations += 1
        else:
            # track which rules have minimal KT distance and do not violate the axioms
            nonviolating_rule_names[Z1.name] += 1

    return possible_violations, total_violations, nonviolating_rule_names


def run_axiom_experiment(output_file="results/axiom_experiment.csv"):
    """
    Run an experiment on all axioms over set parameters, save the results as they become available.
    :return:
    """
    n_profiles = 500
    all_num_voters = [100]
    all_num_cands = [20, 15, 10, 5]
    all_num_splits = [50]


    all_distributions = [
        "MALLOWS-0.4",
        # "MALLOWS-RELPHI-R",
        "URN-R",
        "plackett_luce",
        "single_peaked_conitzer",
        "IC"
    ]
    axioms = [
        # weak_consistency,
        # strong_consistency,
        # homogeneity,
        reversal_symmetry,
        monotonicity,
        consistency,
    ]
    possible_rules = [
        vu.two_approval_ranking,
        vu.borda_ranking,
        vu.plurality_ranking,
        vu.plurality_veto_ranking,
        vu.antiplurality_ranking,
        vu.dowdall_ranking,
    ]

    header = ["n_voters", "n_candidates", "n_splits", "pref_dist", "axiom_name", "possible_violations",
              "total_violations", "non_violating_rule_names"]
    rows = []

    for n_voters, n_cands, n_splits, dist in itertools.product(all_num_voters, all_num_cands, all_num_splits,
                                                               all_distributions):
        kwargs = {"m": n_cands, "k": n_cands}

        profiles = [du.generate_profile(distribution=dist,
                                        num_voters=n_voters,
                                        num_candidates=n_cands) for _ in range(n_profiles)]
        ar = evaluate_axioms(axioms=axioms,
                             profile_set=profiles,
                             possible_rules=possible_rules,
                             n_splits=n_splits, **kwargs)

        row_prefix = [n_voters, n_cands, n_splits, dist]
        for ax, ax_data in ar.items():
            # row = [n_voters, n_cands, n_splits, dist]
            row = row_prefix + [ax, ax_data['possible_violations'], ax_data['total_violations'],
                                str(ax_data['nonviolating_rule_names'])]
            rows.append(row)

        df = pd.DataFrame(columns=header, data=rows)

        df.to_csv(output_file, index=False)
        print(f"Saved row for (n_voters, n_cands, n_splits, dist) = {n_voters, n_cands, n_splits, dist}")


def test_axioms_on_saved_profiles(output_file="results/axiom_experiment-preflib.csv"):
    # n_profiles = 500
    # all_num_voters = [16, 32, 64]
    # all_num_cands = [5, 10, 20]
    min_alternatives, max_alternatives, min_voters, max_voters = 5, 20, 4, 1000
    dist = "preflib"
    profile_path = f"preflib/preflib-n_profiles={1392}-min_alternatives={min_alternatives}-max_alternatives={max_alternatives}-min_voters={min_voters}-max_voters={max_voters}.csv"
    profile_df = pd.read_csv(profile_path)
    all_num_splits = [50]

    all_distributions = [
        "MALLOWS-RELPHI-R",
        "URN-R",
        "plackett_luce",
        "single_peaked_conitzer",
        "IC"
    ]
    axioms = [
        # weak_consistency,
        # strong_consistency,
        # homogeneity,
        reversal_symmetry,
        monotonicity,
        consistency,
    ]
    possible_rules = [vu.plurality_ranking,
                      vu.plurality_veto_ranking,
                      vu.borda_ranking,
                      vu.antiplurality_ranking,
                      # vu.copeland_ranking,
                      # vu.two_approval,
                      vu.dowdall_ranking,
                      ]

    header = ["n_voters", "n_candidates", "n_splits", "pref_dist", "axiom_name", "possible_violations",
              "total_violations", "non_violating_rule_names"]
    rows = []

    # map each num_splits to a dict containing a dict with a row for each axiom
    # track an ongoing record of aggregate stats for each axiom
    # store triple of (possible violations, total violations, Counter)
    # aggregate_results = {ns: {ax.name: [0, 0, Counter()] for ax in axioms} for ns in all_num_splits}
    aggregate_results = {ns: dict() for ns in all_num_splits}

    for n_candidates in range(min_alternatives, max_alternatives+1):

        for n_splits in all_num_splits:
            for row in profile_df.itertuples():
                n_voters = row.n_voters
                if n_candidates != row.n_candidates:
                    # only evaluate matching rows
                    continue

                profile = np.array(eval(row.profile))

                kwargs = {"m": n_candidates, "k": n_candidates}

                ar = evaluate_axioms(axioms=axioms,
                                     profile_set=[profile],
                                     possible_rules=possible_rules,
                                     n_splits=n_splits,
                                     **kwargs)

                row_prefix = [n_voters, n_candidates, n_splits, dist]
                for ax, ax_data in ar.items():
                    # add single new row for this profile
                    row = row_prefix + [ax, ax_data['possible_violations'], ax_data['total_violations'],
                                        str(ax_data['nonviolating_rule_names'])]
                    rows.append(row)

                    # update aggregate statistics
                    if ax not in aggregate_results[n_splits]:
                        aggregate_results[n_splits][ax] = [0, 0, Counter()]
                    aggregate_results[n_splits][ax][0] += ax_data['possible_violations']
                    aggregate_results[n_splits][ax][1] += ax_data['total_violations']
                    ongoing_counter = aggregate_results[n_splits][ax][2]
                    updated_counter = ongoing_counter + ax_data['nonviolating_rule_names']
                    aggregate_results[n_splits][ax][2] = updated_counter

                # # add aggregate rows to the top of all individual rows
                # aggregate_rows = []
                # for ns, ax_dict in aggregate_results.items():
                #     for ax, aggregate_data in ax_dict.items():
                #         aggregate_rows.append([
                #             n_voters, n_candidates, ns, dist, ax, aggregate_data[0], aggregate_data[1], str(aggregate_data[2])
                #         ])
                #
                # all_rows = aggregate_rows + rows

                df = pd.DataFrame(columns=header, data=rows)

                df.to_csv(output_file, index=False)
                print(f"Saved row for (n_voters, n_cands, n_splits, dist) = {n_voters, n_candidates, n_splits, dist}")


if __name__ == "__main__":
    run_axiom_experiment(output_file="results/axiom_experiment.csv")
    test_axioms_on_saved_profiles(output_file="results/axiom_experiment-preflib.csv")
