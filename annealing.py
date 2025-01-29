import pprint

import numpy as np
from simanneal import Annealer
import rule_comparison as rc
import utils.voting_utils as vut
import random
import pref_voting
import utils.data_utils as du
import utils.voting_utils as vu


class ScoreVectorAnnealer(Annealer):

    def __init__(self, initial_state, profiles, n_splits):
        super().__init__(initial_state)
        self.m = len(profiles[0][0])
        self.all_profiles = profiles
        self.n_splits = n_splits

    def energy(self):

        dist = rc.kt_distance_between_many_profiles_with_positional_scoring_rule(profiles=self.all_profiles,
                                                                                 n_splits=self.n_splits,
                                                                                 score_vector=self.state)
        # # get KT dist of current state for many splits
        # all_dists = []
        # for profile in self.all_profiles:
        #     all_splits = rc.make_split_indices(profile, self.n_splits)
        #     for split in all_splits:
        #         s1, s2 = rc.splits_from_split_indices(profile, split_indices=split)
        #         s1 = pref_voting.profiles.Profile(s1)
        #         s2 = pref_voting.profiles.Profile(s2)
        #         ranking1 = vut.positional_scoring_scores(s1, score_vector=self.state)
        #         ranking1 = vut.scores_to_tuple_ranking(ranking1)
        #         ranking2 = vut.positional_scoring_scores(s2, score_vector=self.state)
        #         ranking2 = vut.scores_to_tuple_ranking(ranking2)
        #         dist = rc.kt_distance_between_rankings(ranking1, ranking2)
        #         all_dists.append(dist)
        #
        # mean_kt_dist = np.mean(all_dists)

        return dist

    def move(self):
        """
        Adjust the current score vector slightly. Treat the score vector as the state and slightly modify it with
        :return:
        """
        max_iters = 100
        num_iters = 0
        while num_iters < max_iters:
            num_iters += 1
            # pick an element with room to grow
            index = random.randint(0, self.m - 1)
            if index == 0:
                amount = random.uniform(0.05, 1)
                break
            elif self.state[index] == self.state[index - 1]:
                # no room to increase this position
                continue
            else:
                amount = random.uniform(0, self.state[index - 1] - self.state[index])
                break
            # if (index == 0) or (self.state[index] < self.state[index-1]):
            #     # pick some step size that will not make this element larger than the earlier one
            #     amount = random.uniform(0.05, 1)
            #     if index > 0:
            #         amount = min(amount, self.state[index-1]-self.state[index])
            #     break
        # print(f"Searched {num_iters} iterations to find index {index} which will move by {amount}.")

        try:
            # try, in case amount is not set
            self.state[index] += amount
        except Exception as e:
            print(f"Found exception: {e}. Continuing with no annealing move.")


class SingleProfileScoreVectorAnnealer(Annealer):

    def __init__(self, initial_state, profile, n_splits, m, k):
        super().__init__(initial_state)

        self.m = m
        self.k = k
        self.profile = np.array(profile)
        self.max_iters = 100

        # pre-calculate splits and ranking_counts matrix
        # count number of voters ranking each alternative at each rank
        self.ranking_counts = [[0 for _ in range(self.k)] for _ in range(self.m)]
        # ranking_counts[i][j] is number of voters ranking candidate i in position j
        for ranking in profile:
            for rank, alternative in enumerate(ranking):
                self.ranking_counts[alternative][rank] += 1

        # make ranking count matrix for each split
        self.splits = rc.make_split_indices(profile, n_splits)
        self.all_s1_rank_counts = []
        self.all_s2_rank_counts = []
        for split in self.splits:
            s1, s2 = rc.splits_from_split_indices(self.profile, split_indices=split)

            s1_rank_counts = [[0 for _ in range(len(s1[0]))] for _ in range(m)]
            s2_rank_counts = [[0 for _ in range(len(s2[0]))] for _ in range(m)]

            # make ranking counts for each split
            for ranking in s1:
                for rank, alternative in enumerate(ranking):
                    s1_rank_counts[alternative][rank] += 1
            for ranking in s2:
                for rank, alternative in enumerate(ranking):
                    s2_rank_counts[alternative][rank] += 1

            # save ranking counts for later
            self.all_s1_rank_counts.append(np.array(s1_rank_counts))
            self.all_s2_rank_counts.append(np.array(s2_rank_counts))

    def energy(self):
        all_dists = []
        for s1, s2 in zip(self.all_s1_rank_counts, self.all_s2_rank_counts):

            # s1 is ndarray of rank counts
            # multiply by transposed score vector to get scores

            state = np.array(self.state)
            state = np.transpose(state)

            s1_scores = np.matmul(s1, state)
            s2_scores = np.matmul(s2, state)

            ranking1 = vu.scores_to_tuple_ranking(s1_scores)
            ranking2 = vu.scores_to_tuple_ranking(s2_scores)

            dist = rc.kt_distance_between_rankings(ranking1, ranking2)
            all_dists.append(dist)

        return np.mean(all_dists)

        # all_dists = []
        # for split in self.splits:
        #     s1, s2 = rc.splits_from_split_indices(self.profile, split_indices=split)
        #     s1 = pref_voting.profiles.Profile(s1)
        #     s2 = pref_voting.profiles.Profile(s2)
        #     ranking1 = vu.positional_scoring_scores(s1, score_vector=score_vector)
        #     ranking1 = vu.scores_to_tuple_ranking(ranking1)
        #     ranking2 = vu.positional_scoring_scores(s2, score_vector=score_vector)
        #     ranking2 = vu.scores_to_tuple_ranking(ranking2)
        #     dist = kt_distance_between_rankings(ranking1, ranking2)
        #     all_dists.append(dist)
        #
        # mean_kt_dist = np.mean(all_dists)
        # return mean_kt_dist
        #
        # return dist

    def move(self):
        """
        Adjust the current score vector slightly. Treat the score vector as the state and slightly modify it with
        :return:
        """
        num_iters = 0
        while num_iters < self.max_iters:
            num_iters += 1
            # pick an element with room to grow
            index = random.randint(0, self.k - 1)
            if index == 0:
                amount = random.uniform(0.05, 1)
                break
            elif self.state[index] == self.state[index - 1]:
                # no room to increase this position
                continue
            else:
                amount = random.uniform(0, self.state[index - 1] - self.state[index])
                break
            # if (index == 0) or (self.state[index] < self.state[index-1]):
            #     # pick some step size that will not make this element larger than the earlier one
            #     amount = random.uniform(0.05, 1)
            #     if index > 0:
            #         amount = min(amount, self.state[index-1]-self.state[index])
            #     break
        # print(f"Searched {num_iters} iterations to find index {index} which will move by {amount}.")

        try:
            # try, in case amount is not set
            self.state[index] += amount
        except Exception as e:
            print(f"Found exception: {e}. Continuing with no annealing move.")


def _score_vector_examples(m=10):
    """
    Generate several score vectors corresponding to well known rules and otherwise.
    :param m:
    :return:
    """
    plurality = [1] + [0 for _ in range(m-1)]
    plurality_veto = [1] + [0 for _ in range(m-2)] + [-1]
    veto = [0 for _ in range(m-1)] + [-1]
    borda = [m-idx-1 for idx in range(m)]
    squared_borda = [(m-idx-1)**2 for idx in range(m)]
    cubed_borda = [(m-idx-1)**3 for idx in range(m)]
    two_approval = [1, 1] + [0 for _ in range(m-2)]
    half_approval = [1] + [0.9 if idx < m//2 else 0 for idx in range(m-1)]
    dowdall = [1/(i+1) for i in range(m)]
    geometric_decreasing = [1/(2**i) for i in range(m)]
    if m % 2 == 1:
        half_approval_degrading = [1] + [0.9 for _ in range(m//2)] + [1/(2**(idx+1)) for idx in range(m//2)]
    else:
        half_approval_degrading = [1] + [0.9 for _ in range(m//2-1)] + [1 / (2 ** (idx + 1)) for idx in range(m//2)]

    # all_score_vectors = [plurality, plurality_veto, veto, borda, squared_borda, cubed_borda, two_approval, symmetric,
    #                      symmetric_geometric]
    all_score_vectors = {
        "plurality": plurality,
        "plurality_veto": plurality_veto,
        "veto": veto,
        "borda": borda,
        "squared_borda": squared_borda,
        "cubed_borda": cubed_borda,
        "two_approval": two_approval,
        "half_approval": half_approval,
        "half_approval_degrading": half_approval_degrading,
        "geometric_decreasing": geometric_decreasing,
        "dowdall": dowdall
    }
    return all_score_vectors


def kt_distance_of_vectors(score_vectors, profiles, n_splits):
    """
    Get the KT distance of each vector averaged over the given profiles.
    :param score_vectors: dict mapping name of vector to the vector
    :return:
    """
    results = dict()
    for rule_name, rule_vector in score_vectors.items():
        dist = rc.kt_distance_between_many_profiles_with_positional_scoring_rule(profiles=profiles,
                                                                                 n_splits=n_splits,
                                                                                 score_vector=rule_vector)
        results[rule_name] = round(dist, 4)
    return results

#
# def run_annealing_experiment(dist="plackett_luce", profiles_per_distribution=40):
#     n_profiles = profiles_per_distribution
#     n_voters = 16
#     n_candidates = 10
#     n_splits = 40
#     n_steps = 1000
#
#     # profiles = [du.generate_profile(distribution=dist,
#     #                                 num_voters=n_voters,
#     #                                 num_candidates=n_candidates) for _ in range(n_profiles)]
#     profiles = du.generate_profiles(distribution=dist,
#                                     profiles_per_distribution=n_profiles,
#                                     num_voters=n_voters,
#                                     num_candidates=n_candidates)
#
#     initial_state = [1] + [0 for _ in range(n_candidates - 1)]
#     tsp = ScoreVectorAnnealer(initial_state=initial_state,
#                               profiles=profiles,
#                               n_splits=n_splits)
#
#     tsp.steps = n_steps
#
#     vector, sw = tsp.anneal()
#
#     print("\n")
#     print(f"Results for distribution: {dist}")
#
#     print(f"Best vector found: {du.normalize_score_vector(vector)} with KT distance: {round(sw, 4)}")
#     print(f"Comparing results of annealing with pre-built vectors.")
#
#     score_vectors = _score_vector_examples(m=n_candidates)
#     score_vectors["annealing_result"] = du.normalize_score_vector(vector)
#
#     results = kt_distance_of_vectors(score_vectors=score_vectors,
#                                      profiles=profiles,
#                                      n_splits=n_splits)
#
#     # Sort/format for easy reading of output
#     results = dict(sorted(results.items(), key=lambda item: item[1]))
#     for name, mean_kt_dist in results.items():
#         print(f"{name}: {mean_kt_dist}")
#
#     return du.normalize_score_vector(vector)


if __name__ == "__main__":
    # all_distributions = [
    #     "MALLOWS-RELPHI-R",
    #     "URN-R",
    #     "plackett_luce",
    #     "single_peaked_conitzer",
    #     "IC"
    # ]
    # best_vectors = {}
    # for dist in all_distributions:
    #     best_state = run_annealing_experiment(dist=dist)
    #     best_vectors[dist] = best_state
    #
    # for dist_name, vector in best_vectors.items():
    #     print(f"Best vector for {dist_name} was: {vector}")

    print("Beginning annealing with mixed distribution")
    mixed_distribution = ["MALLOWS-RELPHI-R", "URN-R", "plackett_luce", "single_peaked_conitzer", "IC"]
    best_state = run_annealing_experiment(dist=mixed_distribution, profiles_per_distribution=20)


