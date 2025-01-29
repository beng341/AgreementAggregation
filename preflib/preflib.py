from preflibtools.instances import OrdinalInstance
import os
from pathlib import Path
import utils.data_utils as du
import pref_voting
import pprint
import pandas as pd


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


if __name__ == "__main__":

    min_alternatives, max_alternatives, min_voters, max_voters = 5, 20, 4, 1000
    make_data_file(min_alternatives, max_alternatives, min_voters, max_voters)
