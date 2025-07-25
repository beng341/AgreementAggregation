import copy
import random

import numpy as np
import preflibtools.instances
from preflibtools.instances import OrdinalInstance
import os
import glob
from pathlib import Path
import utils.data_utils as du
import pref_voting
import pprint
import pandas as pd
import utils.voting_utils as vut
from utils.decorator import method_name


def clean_election_profile(profile):
    """
    Ensure that each profile starts labelling candidates from 0.
    Format each profile as a list of tuples. Assume a strict order, can be updated to accommodate weak orders as needed.
    :param profile:
    :return:
    """
    profile = [[cand[0] for cand in order] for order in profile]

    # Find the minimum valued alternative across all order
    min_value = min(min(order) for order in profile)

    # Ensure orders all begin with alternative 0
    profile = [[cand - min_value for cand in order] for order in profile]

    # # Format each order as tuples.
    # profile = [[(cand, ) for cand in order] for order in profile]

    return profile


def load_matching_profiles(folder, min_alternatives, max_alternatives, min_voters, max_voters, max_num_profiles=None):
    """
    Load all preflib instances in the given file and return the instances with a suitable number of alternatives/voters.
    :param folder:
    :param min_alternatives:
    :param max_alternatives:
    :param min_voters:
    :param max_voters:
    :param max_num_profiles
    :return:
    """

    folder = Path(folder)

    instances = []
    profiles = []
    for file_path in folder.glob("*.soc"):
        if max_num_profiles and len(instances) >= max_num_profiles:
            break
        try:
            # Create new election instance for each file
            instance = OrdinalInstance()
            instance.parse_file(str(file_path))

            # Keep only the elections that have the right number of voters/alternatives
            if instance.num_voters < min_voters or instance.num_voters > max_voters:
                continue
            if instance.num_alternatives < min_alternatives or instance.num_alternatives > max_alternatives:
                continue

            instances.append(instance)
            profiles.append(instance.full_profile())
            print(f"Adding election with {instance.num_voters} voters and {instance.num_alternatives} alternatives.")

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return instances


def make_data_file_from_profiles(instances, output_folder, min_alternatives, max_alternatives, min_voters, max_voters):
    """
    Make a data file suitable for training or evaluation, based on the given profiles.
    :param instances: PreflibInstance object
    :param output_folder
    :param num_winners
    :param num_alternatives
    :param axioms
    :return:
    """

    filename = f"preflib-n_profiles={len(instances)}-min_alternatives={min_alternatives}-max_alternatives={max_alternatives}-min_voters={min_voters}-max_voters={max_voters}.csv"

    profile_dict = {"n_voters": [], "n_candidates": [], "profile": []}

    for idx, instance in enumerate(instances):

        # num_alternatives = instance.num_alternatives
        profile = instance.full_profile()
        profile = clean_election_profile(profile)
        # pref_voting_profile = pref_voting.profiles.Profile(rankings=profile)

        profile_dict["profile"].append(profile)
        profile_dict["n_voters"].append(instance.num_voters)
        profile_dict["n_candidates"].append(instance.num_alternatives)

    # Output the complete dataset for good measure, likely redundant
    profiles_df = pd.DataFrame.from_dict(profile_dict)
    filepath = os.path.join(output_folder, filename)
    profiles_df.to_csv(filepath, index=False)
    print(f"Saved filtered preflib data to: {filepath}")


def make_data_file(min_alternatives, max_alternatives, min_voters, max_voters):
    instances = load_matching_profiles(
        folder="preflib/soc",
        min_alternatives=min_alternatives,
        max_alternatives=max_alternatives,
        min_voters=min_voters,
        max_voters=max_voters
    )
    print(f"There are {len(instances)} elections.")

    make_data_file_from_profiles(instances=instances,
                                 output_folder="preflib",
                                 min_alternatives=min_alternatives,
                                 max_alternatives=max_alternatives,
                                 min_voters=min_voters,
                                 max_voters=max_voters)


def load_formula_1_elections():
    """
    Load election data and return profiles along with the winner that occurred in the real election.
    F1 data contains strict orders with no ties and an incomplete ranking over candidates.
    :return:
    """
    # F1 race data comes in two formats: Complete and incomplete. For now, use only the imputed complete preferences.
    # file_pattern = "preflib/useful_data/F1/*.soc"
    file_pattern = "preflib/F1/*.soc"
    from preflibtools.instances import OrdinalInstance

    preference_files = []
    instances = []

    for file in glob.glob(file_pattern):
        instances.append(OrdinalInstance(file))

    # instance names are the years which also correspond to score vectors
    # map instances to their score vectors
    # instance_score_vectors = dict()
    profile_rules = []
    for instance in instances:
        m = instance.num_alternatives
        year = int(instance.title)
        # if 1981 <= year <= 1990:
        #     vec = [9, 6, 4, 3, 2, 1]
        if 1991 <= year <= 2002:
            # vec = [10, 6, 4, 3, 2, 1]
            vec = vut.f1_1991_ranking_vector(m)
        elif 2003 <= year <= 2009:
            # vec = [10, 8, 6, 5, 4, 3, 2, 1]
            vec = vut.f1_2003_ranking_vector(m)
        elif 2010 <= year <= 2018:
            # vec = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
            vec = vut.f1_2010_ranking_vector(m)
        else:
            # Other years exist in the data but we skip them due to quite a few different rules being used
            # See: https://en.wikipedia.org/wiki/List_of_Formula_One_World_Championship_points_scoring_systems
            continue
            # raise ValueError(f"F1 Year is not an expected value: {year}")

        # # zero-pad and truncate to ensure correct number of scores in vector
        # while len(vec) < m:
        #     vec.append(0)
        # if len(vec) > m:
        #     vec = vec[0:m]

        profile = instance.full_profile()
        profile = clean_election_profile(profile)

        # Create function that acts as voting rule which takes only a profile
        @method_name(name=f"F1-{year}", rule_type="positional_scoring", reversible=True, allows_weak_ranking=True)
        def f1_rule(profile, **kwargs):
            # DO NOT USE: There is a subtle Python issue here which results in slightly incorrect score vectors
            # which sometimes lead to incorrect results. This is redundant anyway; use the specific rules
            # in voting_utils for each set of years.
            scores = vut.positional_scoring_scores(profile,
                                                   vec,
                                                   **kwargs)
            ranking = vut.scores_to_tuple_ranking(scores)
            return ranking

        profile_rules.append((f"F1 Seasons ({year})", profile, f1_rule, m))

    # print some statistics about the amount of data here
    pref_orders = [pr[1] for pr in profile_rules]
    num_races = [len(po) for po in pref_orders]
    print(f"# Races Per Year: Min={min(num_races)}, Max={max(num_races)}, Mean={np.mean(num_races)}, Mode={max(num_races, key=num_races.count)}")

    num_racers = [pr[3] for pr in profile_rules]
    print(f"# Drivers Per Year: Min={min(num_racers)}, Max={max(num_racers)}, Mean={np.mean(num_racers)}, Mode={max(num_racers, key=num_racers.count)}")


    return profile_rules


def load_uk_labour_party_leadership_election():
    """
    Create a list containing tuples of (profile, rule being used, num alternatives).
    Only one entry in the list for the UK Labour election.
    :return:
    """

    path = "preflib/useful_data/UK Labour Party Leadership/00030-00000001.soi"
    instance = OrdinalInstance(path)

    @method_name(name=f"UK Labour (IRV)", allows_weak_ranking=True)
    def labour_rule(profile, **kwargs):
        return vut.instant_runoff_with_incomplete_ballots(profile)

    profile = instance.full_profile()
    profile = clean_election_profile(profile)

    return_list = [("UK Labour", profile, labour_rule, instance.num_alternatives)]

    return return_list


def load_city_election_data(max_count=None, max_n_voters=None):
    """

    :return:
    """
    file_pattern = "preflib/City Data/*.toc"
    # file_pattern = "preflib/useful_data/City Data/*.toc"
    from preflibtools.instances import OrdinalInstance

    preference_files = []
    instances = []

    for file in glob.glob(file_pattern):
        instances.append(OrdinalInstance(file))

    # instance names are the years which also correspond to score vectors
    # map instances to their score vectors
    # instance_score_vectors = dict()
    profile_rules = []
    for instance in instances:

        if max_n_voters and instance.num_voters > max_n_voters:
            print(f"Not using {instance.title} due to {instance.num_voters} > {max_n_voters} voters.")
            continue

        m = instance.num_alternatives

        city = instance.title

        # if city != "2008 San Francisco Board of Supervisors - District 4":
        #     continue
        # else:
        #     pass

        profile = instance.full_profile()
        random.shuffle(profile)
        # profile = [list(order) for order in profile]
        # Find the minimum valued alternative across all order
        min_value = min(min(min(order)) for order in profile)
        # Ensure orders all begin with alternative 0
        profile = [[tuple(alt - min_value for alt in tied_alts) for tied_alts in order] for order in profile]
        # profile = clean_election_profile(profile)

        @method_name(name=f"City-{city}", allows_weak_ranking=True)
        def city_rule(profile, **kwargs):
            return vut.instant_runoff_with_incomplete_ballots(profile)

        profile_rules.append((f"City ({city})", profile, city_rule, m))

        if max_count and len(profile_rules) >= max_count:
            break

    return profile_rules


if __name__ == "__main__":

    load_formula_1_elections()

    # load_uk_labour_party_leadership_election()

    load_city_election_data()

    # min_alternatives, max_alternatives, min_voters, max_voters = 5, 20, 4, 1000
    # make_data_file(min_alternatives, max_alternatives, min_voters, max_voters)
