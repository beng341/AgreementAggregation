#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:38:40 2023

@author: eminberker
"""
# True ranking is 1>2>3>4>...>m
from itertools import permutations, combinations
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import choix
import ranky as rk
import time
import top_k_mallows_master.mallows_model as mm
import top_k_mallows_master.mallows_kendall as mk
import top_k_mallows_master.mallows_hamming as mh
import networkx as nx
import math
# import sdopt_tearing_master.grb_lazy as gl
# import sdopt_tearing_master.grb_pcm as gpcm

from sklearn.linear_model import LinearRegression
from agreement_aggregation import position_scoring, kendall_tau_ranking, topological_sort_kahn, ave_ranking_scores, \
    compute_rms, reviewer_split, total_scores, total_scores_all, kendall_tau, kendall_tau_scores, build_pairwise_graph, \
    loss_sigmoid_torch, kemeny_gurobi_lazy, print_from_scores

use_gurobi = True
use_cvxpy = False

if use_gurobi:
    import gurobipy as gp
    from gurobipy import GRB
    from gurobipy import QuadExpr

    from pprint import pprint
    import seaborn as sns

    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()

    import tensorflow_probability as tfp

    sns.reset_defaults()
    # sns.set_style('whitegrid')
    # sns.set_context('talk')
    sns.set_context(context='talk', font_scale=0.7)
    # tfd = tfp.distributions

if use_cvxpy:
    import cvxpy as cp

from tqdm import tqdm
import torch


# -------------------------------------------------------------
# (SINGLE FUNCTION) FOR GENERATING THE REVIEWER-PROPOSAL ASSSINGMENTS:

def generate_matrix(args):  # Generates an matrix corresponding to reviewer/proposal assignments
    (n, m, k, l) = args
    while True:
        # Initialize an n x n matrix filled with zeros
        matrix = np.zeros((m, n), dtype=int)

        for i in range(m):
            # Randomly select k distinct columns
            available = [x for x in range(n) if sum(matrix[:, x]) < l - 1]
            if len(available) < k:
                cols = available + random.sample([x for x in range(n) if sum(matrix[:, x]) == l - 1],
                                                 k - len(available))
            else:
                cols = random.sample(available, k)
            for col in cols:
                matrix[i][col] = 1

        return matrix


# -------------------------------------------------------------
# (VARIOUS FUNCTIONS) FOR GENERATING REVIEWS, GIVEN REVIEWER-PROPOSAL ASSSINGMENTS:

def generate_random_switch(assignments, pr,
                           args):  # Generate reviews with random flips from the absolute truth with probability pr
    (n, m, k, l) = args
    reviews = [[p for p in range(m) if assignments[p, r] == 1] for r in range(n)]

    for r in range(n):
        for _ in range(len(reviews[r])):
            for i in range(len(reviews[r]) - 1):
                if reviews[r][i] < reviews[r][i + 1] and random.random() > pr:
                    prp1 = reviews[r][i]
                    reviews[r][i] = reviews[r][i + 1]
                    reviews[r][i + 1] = prp1
                elif reviews[r][i] > reviews[r][i + 1] and random.random() < pr:
                    prp1 = reviews[r][i]
                    reviews[r][i] = reviews[r][i + 1]
                    reviews[r][i + 1] = prp1
    return reviews


def plackett_luce(assignments, scores, args):  # generates reviews according to PL distribution with the input scores
    (n, m, k, l) = args
    to_review = [[p for p in range(m) if assignments[p, r] == 1] for r in range(n)]
    reviews = [[0] * k for _ in range(n)]
    for r in range(n):
        scores_to_review = [scores[i] for i in to_review[r]]
        ranking = tfp.distributions.PlackettLuce(scores_to_review)
        r_ranking = ranking.sample()
        r_review = [to_review[r][i] for i in r_ranking]
        reviews[r] = r_review
    return reviews


def borda_generate(args,
                   scores=None):  # Generates according to the noise model for which Borda is MLE (from Conitzer and Sandholm)
    # NOTE: Not getting assingments as input since its assuming
    (n, m, k, l) = args  # We ignore k and l for this, since we are only generating full rankings for borda
    if scores is None:
        scores = torch.linspace(1.0, 0.0, steps=m)
    rankings = generate_strict_rankings(m)
    scored_rankings = (m - rankings) ** scores.unsqueeze(0)
    scores_rankings = scored_rankings.prod(dim=1)
    print("scores_rankings", scores_rankings.shape)
    ranking_indices = torch.multinomial(scores_rankings, n, replacement=True)
    final_rankings = rankings[ranking_indices]
    # This previous method does not work since you cannot sample one by one:
    # to_review=torch.arange(m).repeat(n,1)
    # scores_base=m-to_review
    # reviews= torch.zeros(n,m)
    # row_indices = torch.arange(n).view(-1, 1)
    # for p in range(m-1):
    #     borda_score=1-p/(m-1) #the posiitoinal score of the reviewer we are picking now
    #     scores= scores_base**borda_score
    #     outcome=torch.multinomial(scores,1)
    #     reviews[:,p]=outcome[:,0]
    #     scores_base[row_indices,outcome]=0
    # outcome=torch.multinomial(scores_base*1.0,1) #we dont exponentiate on the last review since 0^0=1, but we want the single remaining unranked proposal to appear instead
    # reviews[:,m-1]=outcome[:,0]

    return final_rankings


def mallows_generate(assignments, phi, args):  # Generates partial rankings according to Mallows noise model using RIM
    (n, m, k, l) = args  # We ignore k and l for this, since we are only generating full rankings for borda
    to_review = torch.tensor([[p for p in range(m) if assignments[p, r] == 1] for r in range(n)])
    full_mallows = torch.tensor(mk.sample(m=n, n=k, phi=phi))
    row_indices = torch.arange(n).unsqueeze(1).expand(-1, l)
    C = to_review[row_indices, full_mallows]

    return C


# -------------------------------------------------------------
# HELPER FUNCTIONS AFTER REVIEWS ARE GENERATED:


def dfs(node, visited, stack, adj_matrix):
    visited[node] = True
    for i in range(len(adj_matrix)):
        if adj_matrix[node, i] and not visited[i]:
            dfs(i, visited, stack, adj_matrix)
    stack.append(node)


def topological_sort_dfs(adj_matrix):  # topologically sorts a DAG, ChatGPT generated.
    n = adj_matrix.shape[0]
    visited = [False] * n
    stack = []

    for i in range(n):
        if not visited[i]:
            dfs(i, visited, stack, adj_matrix)

    return stack[::-1]  # Return reversed stack


def compute_min_feedback_arc_set(adj_matrix):
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    # Compute the minimum feedback arc set
    fas = nx.minimum_feedback_arc_set(G)

    return fas


def count_permutations(tensor):
    # Ensure the tensor is of correct shape (N, 3)
    assert tensor.shape[1] == 3, "Tensor must be of shape (N, 3)"

    # Initialize a dictionary to count occurrences of each permutation
    permutation_counts = {
        (0, 1, 2): 1,
        (0, 2, 1): 1,
        (1, 0, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
        (2, 1, 0): 1,
    }

    # Iterate over each row in the tensor
    for row in tensor:
        # Convert the row to a tuple
        row_tuple = tuple(row.tolist())

        # Increment the count for this permutation
        if row_tuple in permutation_counts:
            permutation_counts[row_tuple] += 1
        else:
            # Handle unexpected permutations if necessary
            pass

    return permutation_counts


def sigmoid_error(M, weights, a, args):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    error = 0
    total_score_1, total_score_2 = total_scores(M, a, args)
    for i in range(l):
        for j in range(i + 1, l):
            error += weights[i] * weights[j] * sigmoid(
                -(total_score_1[i] - total_score_1[j]) * (total_score_2[i] - total_score_2[j]))
    return error


def generate_rankings(n):  # including rankings with ties
    def generate(current, next_number):
        if next_number == n:
            return [current]
        results = []
        # Add next_number as a new rank
        for i in range(len(current) + 1):
            new_ranking = current[:i] + [[next_number]] + current[i:]
            results.extend(generate(new_ranking, next_number + 1))
        # Add next_number as a tie to existing ranks
        for i in range(len(current)):
            new_ranking = current[:i] + [current[i] + [next_number]] + current[i + 1:]
            results.extend(generate(new_ranking, next_number + 1))
        return results

    return generate([], 0)


def generate_combinations(n):
    def generate(current, total):
        if total == n:
            results.append(current)
            return
        if total > n:
            return

        for i in range(1, n + 1):
            generate(current + [i], total + i)

    results = []
    generate([], 0)
    return results


def fact(x):
    if x == 0:
        return 1
    else:
        return fact(x - 1) * x


def total_n_rankings(n):
    output = 0
    all_combinations = generate_combinations(n)
    for combination in all_combinations:
        temp_output = fact(n)
        for i in combination:
            temp_output /= fact(i)
        output += temp_output
    return output


def generate_strict_rankings(n):
    return torch.tensor(list(permutations(range(n))))


def ranking_inverse(ranking):
    inverse = [0 for _ in range(len(ranking))]
    for i in range(len(ranking)):
        inverse[ranking[i]] = [i]
    return inverse


def sorted_rankings(rn, rankings, weights, args):
    (n, m, k, l) = args
    kendall_taus = np.zeros((rn, rn))
    for i in (range(rn)):
        for j in range(i, rn):
            ranking1 = rankings[i]
            ranking2 = rankings[j]
            a = kendall_tau_ranking(weights, ranking1, ranking2, m)
            kendall_taus[i][j] = a
            kendall_taus[j][i] = a

    sorted_kt = np.sort(kendall_taus, axis=None)
    sorted_indices = np.unravel_index(np.argsort(kendall_taus, axis=None), kendall_taus.shape)
    return sorted_kt, sorted_indices


# -------------------------------------------------------------
# (VARIOUS FUNCTIONS) FOR LEARNING THE BEST RULE, GIVEN THE REVIEWER SPLIT:


def loss_step_gurobi(M, weights, args, nabla):
    try:
        # Create a new model
        z = gp.Model("mip1")
        # Create variables
        a = [0 for _ in range(l)]
        a[0] = 1
        a[l - 1] = 0
        for i in range(1, l - 1):
            a[i] = z.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a[" + str(i) + "]"))
            z.addConstr(a[i] + nabla - a[i - 1] <= 0, ("c" + str(i)))
        z.addConstr(a[l - 2] >= a[l - 1] + nabla, ("c" + str(l - 1)))
        total_score_1, total_score_2 = total_scores(M, a, args)

        p = [[0] * m for _ in range(m)]  # indicator variable of whether the disagreement is positive
        t = [[0] * m for _ in range(m)]  # indicator variable of whether the disagreement is zero (there is a tie).
        D = [[0] * m for _ in range(m)]  # disagreement (there is a tie).

        for i in range(m):
            for j in range(i + 1, m):
                p[i][j] = z.addVar(0.0, 1.0, vtype=GRB.BINARY, name=("p[" + str(i) + ',' + str(j) + "]"))
                t[i][j] = z.addVar(0.0, 1.0, vtype=GRB.BINARY, name=("t[" + str(i) + ',' + str(j) + "]"))
                D[i][j] = z.addVar(-1.0, 1.0, vtype=GRB.CONTINUOUS, name=("D[ " + str(i) + ',' + str(j) + "]"))

                z.addConstr(
                    p[i][j] >= -(total_score_1[i] - total_score_1[j]) * (total_score_2[i] - total_score_2[j]) / (
                            l ** 2))  # to indicate whether the disgareement is positive

                z.addConstr(
                    D[i][j] == -(total_score_1[i] - total_score_1[j]) * (total_score_2[i] - total_score_2[j]) / (
                            l ** 2))  # Disagreement

                z.addConstr(-D[i][j] * D[i][j] + nabla ** 4 <= t[i][j])

        z.setObjective(
            sum(weights[i] * weights[j] * (p[i][j] + t[i][j] / 2) for i in range(m) for j in range(i + 1, m)),
            GRB.MINIMIZE)
        z.params.NonConvex = 2
        z.write("out.lp")
        z.optimize()
        for v in z.getVars():
            print('%s %g' % (v.VarName, v.X), end="; ")
        print('Obj: %g' % z.ObjVal)
    except gp.GurobiError as e:
        print('Error code' + str(e.errno) + ': ' + str(e))


    except AttributeError:
        print('Encountered an attribute error')


def loss_relu_gurobi(M, weights, args):
    try:
        # Create a new model
        z = gp.Model("mip1")
        # Create variables
        a = [0 for _ in range(l)]
        a[0] = 1
        a[l - 1] = 0
        for i in range(1, l - 1):
            a[i] = z.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a[" + str(i) + "]"))
            z.addConstr(a[i] - a[i - 1] <= 0, ("c" + str(i)))
        z.addConstr(a[l - 2] >= a[l - 1], ("c" + str(l - 1)))
        total_score_1, total_score_2 = total_scores(M, a, args)

        d = [[0] * m for _ in range(m)]  # to host the disgreement costs
        for i in range(m):
            for j in range(i + 1, m):
                d[i][j] = z.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("d[" + str(i) + ',' + str(j) + "]"))
                z.addConstr(d[i][j] >= -weights[i] * weights[j] * (total_score_1[i] - total_score_1[j]) * (
                        total_score_2[i] - total_score_2[j]))
        z.setObjective(sum(d[i][j] for i in range(m) for j in range(i + 1, m)), GRB.MINIMIZE)
        z.params.NonConvex = 2
        z.write("out.lp")
        z.optimize()
        for v in a[1:-1]:
            print('%s %g' % (v.VarName, v.X), end="; ")

        # for v in z.getVars():
        #     print('%s %g' % (v.VarName, v.X), end="; ")
        print('Obj: %g' % z.ObjVal)
    except gp.GurobiError as e:
        print('Error code' + str(e.errno) + ': ' + str(e))


    except AttributeError:
        print('Encountered an attribute error')


def loss_sigmoid_gurobi(M, weights, args, nabla, steep):
    try:
        # Create a new model
        z = gp.Model("mip1")
        # Create variables
        a = [0 for _ in range(l)]
        a[0] = 1
        a[l - 1] = 0
        for i in range(1, l - 1):
            a[i] = z.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a[" + str(i) + "]"))
            z.addConstr(a[i] + nabla - a[i - 1] <= 0, ("c" + str(i)))
        z.addConstr(a[l - 2] >= a[l - 1] + nabla, ("c" + str(l - 1)))
        total_score_1, total_score_2 = total_scores(M, a, args)

        exp = [[0] * m for _ in range(m)]  # to host the exponential constrains
        p = [[0] * m for _ in range(m)]  # to host the power of each exponent
        sig = [[0] * m for _ in range(m)]  # to host the sigmoid

        for i in range(m):
            for j in range(i + 1, m):
                p[i][j] = z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                   name=("p[" + str(i) + ',' + str(j) + "]"))
                z.addConstr(p[i][j] == weights[i] * weights[j] * (total_score_1[i] - total_score_1[j]) * (
                        total_score_2[i] - total_score_2[
                    j]) / steep)  # this is what we would like to have in our exponent
                exp[i][j] = z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                     name=("exp[" + str(i) + ',' + str(j) + "]"))
                z.addGenConstrExp(p[i][j], exp[i][j])  # ensures exp[i][j]= exp(p[i][j])
                sig[i][j] = z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                     name=("sig[" + str(i) + ',' + str(j) + "]"))
                z.addConstr(sig[i][j] * (1 + exp[i][j]) >= 1)  # ensures sig[i][j] >= 1/ (1+ exp(p[i][j]) )

        z.setObjective(sum(sig[i][j] for i in range(m) for j in range(i + 1, m)), GRB.MINIMIZE)
        z.params.NonConvex = 2
        z.write("out.lp")
        z.optimize()
        for v in a[1:-1]:
            print('%s %g' % (v.VarName, v.X), end="; ")

        # for v in z.getVars():
        #     print('%s %g' % (v.VarName, v.X), end="; ")
        print('Obj: %g' % z.ObjVal)
    except gp.GurobiError as e:
        print('Error code' + str(e.errno) + ': ' + str(e))


    except AttributeError:
        print('Encountered an attribute error')


def loss_sigmoid_cvx(M, weights, args, nabla, steep):
    try:
        # Create variables
        a = [0 for _ in range(l)]
        a[0] = 1
        a[l - 1] = 0
        constraints = []
        for i in range(1, l - 1):
            a[i] = cp.Variable(pos=True)
            constraints += [a[i] + nabla - a[i - 1] <= 0]
        constraints += [a[l - 2] >= a[l - 1] + nabla]
        total_score_1, total_score_2 = total_scores(M, a, args)
        d = [[0] * m for _ in range(m)]  # to host the errors

        for i in range(l):
            for j in range(i + 1, l):
                d[i][j] = cp.Variable(pos=True)
                constraints += [d[i][j] >= cp.multiply(a[i], a[j])]

        #             p[i][j]=z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("p[" +str(i) +',' + str(j)+"]" ))
        #             z.addConstr(p[i][j]==  weights[i]*weights[j]*(total_score_1[i]-total_score_1[j])*(total_score_2[i]-total_score_2[j])/steep) #this is what we would like to have in our exponent
        #             exp[i][j]=z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("exp[" +str(i) +',' + str(j)+"]" ))
        #             z.addGenConstrExp(p[i][j], exp[i][j]) #ensures exp[i][j]= exp(p[i][j])
        #             sig[i][j]=z.addVar(-GRB.INFINITY, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("sig[" +str(i) +',' + str(j)+"]" ))
        #             z.addConstr( sig[i][j]*(1+exp[i][j]) >=1 ) #ensures sig[i][j] >= 1/ (1+ exp(p[i][j]) )

        #     z.setObjective( sum( sig[i][j] for i in range(m) for j in range(i+1,m)), GRB.MINIMIZE)
        #     z.params.NonConvex=2
        #     z.write("out.lp")
        #     z.optimize()
        #     for v in a[1:-1]:
        #         print('%s %g' % (v.VarName, v.X), end="; ")

        #     # for v in z.getVars():
        #     #     print('%s %g' % (v.VarName, v.X), end="; ")
        #     print('Obj: %g' % z.ObjVal)

        obj = cp.Minimize(sum(a[i] for i in range(l)))
        prob = cp.Problem(obj, constraints)
        prob.solve(gp=True)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        for i in range(1, l - 1):
            print("a" + str(i) + ":", a[i].value)

    except AttributeError:
        print('Encountered an attribute error')


def kemeny_gurobi(reviews, args, time_out=None, printout_mode=False):
    """Kemeny-Young optimal rank aggregation"""
    (n, m, k, l) = args

    # maximize c.T * x
    edge_weights = build_pairwise_graph(reviews, args)
    try:
        # Create a new model
        # Create a new model
        m = gp.Model("Minimum_Feedback_Arc_Set")
        if not printout_mode:
            m.Params.LogToConsole = 0
        if not time_out is None:
            m.setParam('TimeLimit', time_out)

        # Create variables
        x = {}
        if printout_mode:
            print("Starting setting variables")
            for i in tqdm(range(n)):
                for j in range(n):
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

            print("Done with variables!")
        else:
            for i in range(n):
                for j in range(n):
                    x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        # Set objective
        if printout_mode:
            print("Starting setting objective")

        m.setObjective(gp.quicksum(edge_weights[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
        if printout_mode:
            print("Done with objective!")

        # Add constraints
        if printout_mode:
            print("Starting setting constraints")
            for i in tqdm(range(n)):
                for j in range(i + 1, n):
                    m.addConstr(x[i, j] + x[j, i] == 1)
                    for k in range(j + 1, n):
                        m.addConstr(x[i, j] + x[j, k] + x[k, i] >= 1)
                        m.addConstr(x[i, k] + x[k, j] + x[j, i] >= 1)
            print("Done with constraints!")
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    m.addConstr(x[i, j] + x[j, i] == 1)
                    for k in range(j + 1, n):
                        m.addConstr(x[i, j] + x[j, k] + x[k, i] >= 1)
                        m.addConstr(x[i, k] + x[k, j] + x[j, i] >= 1)

        # Optimize model
        m.optimize()
        obj = m.getObjective()

        # Extract solution
        # fas = [(i, j) for i in range(n) for j in range(n) if x[i, j].x == 1]
        final_scores = np.array([sum(1 for i in range(n) if x[i, j].x == 1) for j in range(n)])
        return final_scores, obj.getValue()



    except gp.GurobiError as e:
        print('Error code' + str(e.errno) + ': ' + str(e))


    except AttributeError:
        print('Encountered an attribute error')


# -------------------------------------------------------------
# EXPERIMENTS COMPARING VARIOUS METHODS:


def adam_vs_sgd(n_trials, args, sensitivity=0.001):
    (n, m, k, l) = args
    adam_errors = np.zeros(n_trials)
    sgd_errors = np.zeros(n_trials)

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    for i in tqdm(range(n_trials)):
        assignments = all_assignments[i]
        # reviews=generate_reviews(assignments,pr,args)
        alpha = 0.5
        scores = [np.exp(alpha * i) for i in range(m, 0, -1)]
        reviews = plackett_luce(assignments, scores, args)

        reviewers, M, weights = reviewer_split(reviews, args, 2)

        nabla = 0.001
        min_steps = 100
        max_steps = 5000
        steep = 500

        a_torch_adam, torch_losses_adam = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps,
                                                             max_steps, lr=1e-1, print_mode=True, opt_type="Adam")
        a_torch_sgd, torch_losses_sgd = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps,
                                                           max_steps, lr=1e-1, print_mode=True, opt_type="SGD")

        kt_adam, adam_disagreements, adam_ties = kendall_tau(M, weights, a_torch_adam, args, nabla)
        adam_errors[i] = kt_adam

        kt_sgd, sgd_disagreements, sgd_ties = kendall_tau(M, weights, a_torch_sgd, args, nabla)
        print("kt_sgd", kt_sgd)
        sgd_errors[i] = kt_sgd

    return adam_errors, sgd_errors


def pl_vs_torch(n_trials, args, sensitivity=0.001):
    (n, m, k, l) = args
    mle_errors = np.zeros(n_trials)
    torch_errors = np.zeros(n_trials)

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    for i in tqdm(range(n_trials)):
        assignments = all_assignments[i]
        # reviews=generate_reviews(assignments,pr,args)
        alpha = 0.5
        scores = [np.exp(alpha * i) for i in range(m, 0, -1)]
        reviews = plackett_luce(assignments, scores, args)

        reviewers, M, weights = reviewer_split(reviews, args, 2)

        data1 = [reviews[i] for i in reviewers[0]]
        data2 = [reviews[i] for i in reviewers[1]]

        scores1 = choix.opt_rankings(m, data1)
        scores2 = choix.opt_rankings(m, data2)

        mle_error, mle_disagreements, mle_ties = kendall_tau_scores(weights, scores1, scores2, args)

        mle_errors[i] = mle_error

        nabla = 0.001
        min_steps = 100
        max_steps = 5000
        steep = 500

        a_torch_temp, torch_losses = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps, max_steps,
                                                        lr=1e-1)
        kt_torch, torch_disagreements, torch_ties = kendall_tau(M, weights, a_torch_temp, args, nabla)

        torch_errors[i] = kt_torch
    return mle_errors, torch_errors


def brute_vs_torch(n_trials, args, nabla1=0.01, nabla2=1e-6, nabla=0.001, min_steps=100, max_steps=5000, time_out=None):
    (n, m, k, l) = args
    brute_errors = np.zeros(n_trials)
    torch_errors = np.zeros(n_trials)

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    for f in tqdm(range(n_trials)):
        assignments = all_assignments[f]
        # reviews=generate_reviews(assignments,pr,args)
        alpha = 0.5
        scores = [np.exp(alpha * i) for i in range(m, 0, -1)]
        reviews = plackett_luce(assignments, scores, args)

        reviewers, M, weights = reviewer_split(reviews, args, 2)
        kt_torch = float('inf')

        for steep in [10, 50, 100, 200, 300, 400]:
            for lr in [1e-1, 1e-2]:
                sensitivity = 0.000000000001
                a_torch_temp, torch_losses = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps,
                                                                max_steps, lr=lr, print_mode=False)
                kt_torch_temp, dis, tie = kendall_tau(M, weights, a_torch_temp, args, nabla)
                if kt_torch_temp < kt_torch:
                    kt_torch = kt_torch_temp
                    a_torch = a_torch_temp

        torch_errors[f] = kt_torch

        rankings = generate_rankings(m)
        min_kt = float('inf')
        rn = len(rankings)

        # sorted_kt,sorted_indices = sorted_rankings(rn, rankings, weights, args)
        # final=np.where(sorted_kt==torch_error)[0][-1]+2

        # Before of checking every combination, we check for each ranking for the two sets, to rule out "infeasible" rankings for bot sets
        # Main idea: if set_1 reviews cant fit into ranking_1 for any a, they wont be able to do so for any ranking_2 for set 2 (more constraints)

        achieveable_rankings_1 = [True for _ in range(rn)]
        achieveable_rankings_2 = [True for _ in range(rn)]

        for i in (range(rn)):
            current_ranking = rankings[i]
            try:
                # Create a new model
                z1 = gp.Model("mip1")  # check if set 1 is compatible with current_ranking
                z2 = gp.Model("mip2")  # check if set 2 is compatible with current_ranking

                z1.Params.LogToConsole = 0
                z2.Params.LogToConsole = 0
                if not time_out is None:
                    z1.setParam('TimeLimit', time_out)
                    z2.setParam('TimeLimit', time_out)

                z1.update()
                z2.update()

                # Create variables
                a1 = [0 for _ in range(l)]
                a1[0] = 1
                a1[l - 1] = 0

                a2 = [0 for _ in range(l)]
                a2[0] = 1
                a2[l - 1] = 0

                for j in range(1, l - 1):
                    a1[j] = z1.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a1[" + str(j) + "]"))
                    a2[j] = z2.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a2[" + str(j) + "]"))

                    # z.addConstr( a[j]==a3[j] , ("c"+str(i)) ) 

                    z1.addConstr(a1[j] - a1[j - 1] <= 0, ("c1" + str(i)))
                    z2.addConstr(a2[j] - a2[j - 1] <= 0, ("c2" + str(i)))

                z1.addConstr(a1[l - 2] >= a1[l - 1], ("c1" + str(l - 1)))
                z2.addConstr(a2[l - 2] >= a2[l - 1], ("c2" + str(l - 1)))
                print("MM", M)
                print("mm", M[0][:][2])
                check = [sum(a1[i] * M[0][i, 2] for i in range(k))]
                print("Check", check)

                total_score_11, total_score_12 = total_scores(M, a1, args)
                total_score_21, total_score_22 = total_scores(M, a2, args)

                b1 = [0 for _ in range(m)]
                b2 = [0 for _ in range(m)]

                for j in range(m):
                    b1[j] = z1.addVar(0.0, l, vtype=GRB.CONTINUOUS, name=("b1[" + str(j) + "]"))
                    b2[j] = z2.addVar(0.0, l, vtype=GRB.CONTINUOUS, name=("b2[" + str(j) + "]"))

                for j in range(len(current_ranking)):
                    for t in range(len(current_ranking[j]) - 1):
                        z1.addConstr(b1[current_ranking[j][t]] == b1[current_ranking[j][t + 1]])
                        z2.addConstr(b2[current_ranking[j][t]] == b2[current_ranking[j][t + 1]])
                    if j < len(current_ranking) - 1:
                        z1.addConstr(b1[current_ranking[j][-1]] >= b1[current_ranking[j + 1][0]] + nabla1)
                        z2.addConstr(b2[current_ranking[j][-1]] >= b2[current_ranking[j + 1][0]] + nabla1)

                z1.setObjective(sum((b1[i] - total_score_11[i]) ** 2 for i in range(m)), GRB.MINIMIZE)
                z2.setObjective(sum((b2[i] - total_score_22[i]) ** 2 for i in range(m)), GRB.MINIMIZE)

                z1.params.NonConvex = 2
                z2.params.NonConvex = 2

                z1.write("out1.lp")
                z2.write("out2.lp")

                z1.optimize()
                z2.optimize()

                # for v in a[1:-1]:
                #     print('%s %g' % (v.VarName, v.X), end="; ")

                # for v in z.getVars():
                #     print('%s %g' % (v.VarName, v.X), end="; ")
                if z1.ObjVal > nabla2:
                    achieveable_rankings_1[i] = False
                if z2.ObjVal > nabla2:
                    achieveable_rankings_2[i] = False
            except gp.GurobiError as e:
                print('Error code' + str(e.errno) + ': ' + str(e))

            except AttributeError:
                print('Encountered an attribute error')

        # kt_brute, a_brute= brute_force_old(final,sorted_kt, sorted_indices, args)
        # brute_errors[f]=kt_brute

        kt_brute = float('inf')
        for r1 in np.where(achieveable_rankings_1)[0]:
            ranking1 = rankings[r1]
            for r2 in np.where(achieveable_rankings_2)[0]:
                ranking2 = rankings[r2]
                ranking_kt = kendall_tau_ranking(weights, ranking1, ranking2, n)
                if ranking_kt < kt_brute:
                    try:
                        # Create a new model
                        z = gp.Model("mip")
                        z.Params.LogToConsole = 0
                        if not time_out is None:
                            z.setParam('TimeLimit', time_out)

                        # Create variables
                        a = [0 for _ in range(l)]
                        a[0] = 1
                        a[l - 1] = 0
                        # a3=[1,0.05150001581190617,0.05150001581190617,0]
                        for j in range(1, l - 1):
                            a[j] = z.addVar(0.0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=("a[" + str(j) + "]"))
                            # z.addConstr( a[j]==a3[j] , ("c"+str(i)) ) 

                            z.addConstr(a[j] - a[j - 1] <= 0, ("c" + str(i)))
                        z.addConstr(a[l - 2] >= a[l - 1], ("c" + str(l - 1)))
                        total_score_1, total_score_2 = total_scores(M, a, args)

                        b1 = [0 for _ in range(m)]
                        b2 = [0 for _ in range(m)]

                        for j in range(m):
                            b1[j] = z.addVar(0.0, l, vtype=GRB.CONTINUOUS, name=("b1[" + str(j) + "]"))
                            b2[j] = z.addVar(0.0, l, vtype=GRB.CONTINUOUS, name=("b2[" + str(j) + "]"))

                        for j in range(len(ranking1)):
                            for t in range(len(ranking1[j]) - 1):
                                z.addConstr(b1[ranking1[j][t]] == b1[ranking1[j][t + 1]])
                            if j < len(ranking1) - 1:
                                z.addConstr(b1[ranking1[j][-1]] >= b1[ranking1[j + 1][0]] + nabla1)

                        for j in range(len(ranking2)):
                            for t in range(len(ranking2[j]) - 1):
                                z.addConstr(b2[ranking2[j][t]] == b2[ranking2[j][t + 1]])
                            if j < len(ranking2) - 1:
                                z.addConstr(b2[ranking2[j][-1]] >= b2[ranking2[j + 1][0]] + nabla1)

                        z.setObjective(
                            sum((b1[i] - total_score_1[i]) ** 2 + (b2[i] - total_score_2[i]) ** 2 for i in range(m)),
                            GRB.MINIMIZE)
                        z.params.NonConvex = 2
                        z.write("out.lp")
                        z.optimize()
                        # for v in a[1:-1]:
                        #     print('%s %g' % (v.VarName, v.X), end="; ")

                        # for v in z.getVars():
                        #     print('%s %g' % (v.VarName, v.X), end="; ")
                        if z.ObjVal <= nabla2:
                            kt_brute = ranking_kt
                            a_brute = [0 for i in range(l)]
                            a_brute[0] = 1
                            counter = 1
                            for v in a[1:-1]:
                                a_brute[counter] = v.X
                                # print('%s %g' % (v.VarName, v.X), end="; ")
                                counter += 1
                        # print('Obj: %g' % z.ObjVal)
                    except gp.GurobiError as e:
                        print('Error code' + str(e.errno) + ': ' + str(e))


                    except AttributeError:
                        print('Encountered an attribute error')
        brute_errors[f] = kt_brute
    return brute_errors, torch_errors


def borda_vs_torch(n_trials, args, nabla1=0.01, nabla2=1e-6, sensitivity=0.001):
    (n, m, k, l) = args
    borda_errors = np.zeros(n_trials)
    torch_errors = np.zeros(n_trials)

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    a_borda = torch.linspace(1.0, 0.0, steps=k)
    for i in tqdm(range(n_trials)):
        assignments = all_assignments[i]
        # reviews=generate_reviews(assignments,pr,args)

        reviews = borda_generate(args)

        reviewers, M, weights = reviewer_split(reviews, args, 2)
        nabla = 0.001

        kt_borda, borda_disagreements, borda_ties = kendall_tau(M, weights, a_borda, args, nabla)

        borda_errors[i] = kt_borda

        min_steps = 100
        max_steps = 5000
        steep = 500

        a_torch_temp, torch_losses = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps, max_steps,
                                                        lr=1e-1)
        kt_torch, torch_disagreements, torch_ties = kendall_tau(M, weights, a_torch_temp, args, nabla)

        torch_errors[i] = kt_torch
    return borda_errors, torch_errors


def pl_vs_borda(n_trials, args, gen_pl=True,
                nabla=0.001):  # tests agreements for PL MLE vs borda, when data is generated according to PL if gen_PL and according to borda otherwise
    (n, m, k, l) = args
    mle_errors = np.zeros(n_trials)
    borda_errors = np.zeros(n_trials)

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    alpha = 0.5
    scores = [np.exp(alpha * i) for i in range(m, 0, -1)]
    if not gen_pl:
        n_samples = n * n_trials
        args2 = (n_samples, m, k, l)
        all_reviews = borda_generate(args2)

    for i in tqdm(range(n_trials)):
        assignments = all_assignments[i]
        # reviews=generate_reviews(assignments,pr,args)
        if gen_pl:
            reviews = plackett_luce(assignments, scores, args)
        else:
            reviews = all_reviews[i * n:(i + 1) * n]

        reviewers, M, weights = reviewer_split(reviews, args, 2)

        data1 = [reviews[j] for j in reviewers[0]]
        data2 = [reviews[j] for j in reviewers[1]]

        scores1 = choix.opt_rankings(m, data1)
        scores2 = choix.opt_rankings(m, data2)
        # print("pl scores", scores1,scores2)
        mle_error, mle_disagreements, mle_ties = kendall_tau_scores(weights, scores1, scores2, args)

        mle_errors[i] = mle_error

        a_borda = torch.linspace(1.0, 0.0, steps=k)
        kt_borda, borda_disagreements, borda_ties = kendall_tau(M, weights, a_borda, args, nabla)
        # print("borda scores", total_scores(M, a_borda, args))

        borda_errors[i] = kt_borda
    return mle_errors, borda_errors


def general_experiment(n_trials, args, generator, rules=[0, 0, 0, 0, 0, 0], alpha=0.5, phi=0.4, min_steps=100,
                       max_steps=5000, steep=300, sensitivity=0.001, lr=1e-1, nabla=0.001):
    assert isinstance(n_trials, int)
    assert generator in ["PL", "Borda", "Mallows"]  # Which generator will be used for the data?
    assert isinstance(rules, list) and len(
        rules) == 6  # Which rules will be tested on? 0-1 vector encoding [PL MLE, Borda, Mallows MLE (Kemeny), Torch Sigmoid]
    # phi is the mallows parameter, alpha is the pl score parameter
    # min/max step, sensitivity, and steep are torch sigmoid paramters
    # nabla is a sensistivity parameter for kendall_tau helper function

    (n, m, k, l) = args

    pl_mle_errors = np.zeros(n_trials)
    borda_errors = np.zeros(n_trials)
    kemeny_errors = np.zeros(n_trials)
    torch_errors = np.zeros(n_trials)
    min_max_errors = np.zeros(n_trials)
    rms_errors = np.zeros(n_trials)

    pl_scores = [np.exp(alpha * i) for i in range(m, 0, -1)]

    all_assignments = []

    while (len(all_assignments) < n_trials):
        try:
            assignments = generate_matrix(args)
            all_assignments.append(assignments)
        except ValueError:
            print("skipped")

    if generator == "Borda":
        n_samples = n * n_trials
        args2 = (n_samples, m, k, l)
        borda_all_reviews = borda_generate(args2)

    for i in tqdm(range(n_trials)):
        assignments = all_assignments[i]
        # reviews=generate_reviews(assignments,pr,args)
        if generator == "PL":
            reviews = plackett_luce(assignments, pl_scores, args)
        elif generator == "Borda":
            reviews = borda_all_reviews[i * n:(i + 1) * n]
        else:
            reviews = mallows_generate(assignments, phi, args)

        reviewers, M, weights = reviewer_split(reviews, args, 2)
        if rules[0] == 1 or rules[2] == 1:
            data1 = [reviews[j] for j in reviewers[0]]
            data2 = [reviews[j] for j in reviewers[1]]

        if rules[0] == 1:  # testing for PL MLE
            scores1 = choix.opt_rankings(m, data1)
            scores2 = choix.opt_rankings(m, data2)
            # print("pl scores", scores1,scores2)
            mle_error, mle_disagreements, mle_ties = kendall_tau_scores(weights, scores1, scores2, args)
            pl_mle_errors[i] = mle_error

        if rules[1] == 1:  # testing for BORDA
            a_borda = torch.linspace(1.0, 0.0, steps=k)
            kt_borda, borda_disagreements, borda_ties = kendall_tau(M, weights, a_borda, args, nabla)
            # print("borda scores", total_scores(M, a_borda, args))
            borda_errors[i] = kt_borda

        if rules[2] == 1:  # testing for KEMENY (MALLOWS MLE)
            kemeny_scores_1 = kemeny_gurobi(data1, args)
            kemeny_scores_2 = kemeny_gurobi(data2, args)
            kemeny_error, kemeny_disagreements, kemeny_ties = kendall_tau_scores(weights, kemeny_scores_1,
                                                                                 kemeny_scores_2, args)
            kemeny_errors[i] = kemeny_error

        if rules[3] == 1:  # testing for TORCH SIGMOID
            a_torch_temp, torch_losses = loss_sigmoid_torch(M, weights, args, 0, steep, sensitivity, min_steps,
                                                            max_steps, lr=lr, print_mode=False)
            kt_torch, torch_disagreements, torch_ties = kendall_tau(M, weights, a_torch_temp, args, nabla)
            torch_errors[i] = kt_torch

        if rules[4] == 1 or rules[5] == 1:  # testing for average Borda with SOME rejection:
            proposal_ranks1 = [[] for _ in range(m)]  # for each proposal, we will add its ranks here. (REVIEWERS 1)
            proposal_ranks2 = [[] for _ in range(m)]  # for each proposal, we will add its ranks here. (REVIEWERS 2)
            for f in range(n):
                if f in reviewers[0]:
                    for j in range(k):
                        proposal_ranks1[reviews[f][j]].append(j)
                elif f in reviewers[1]:
                    for j in range(k):
                        proposal_ranks2[reviews[f][j]].append(j)
        if rules[4] == 1:
            scores1 = ave_ranking_scores(proposal_ranks1, max_mix_rejection=True, rms_rejection=False)
            scores2 = ave_ranking_scores(proposal_ranks2, max_mix_rejection=True, rms_rejection=False)
            min_max_error, min_max_disagreements, min_max_ties = kendall_tau_scores(weights, scores1, scores2, args)
            min_max_errors[i] = min_max_error

        if rules[5] == 1:
            scores1 = ave_ranking_scores(proposal_ranks1, max_mix_rejection=False, rms_rejection=True)
            scores2 = ave_ranking_scores(proposal_ranks2, max_mix_rejection=False, rms_rejection=True)
            rms_error, rms_disagreements, rms_ties = kendall_tau_scores(weights, scores1, scores2, args)
            rms_errors[i] = rms_error

    return pl_mle_errors, borda_errors, kemeny_errors, torch_errors, min_max_errors, rms_errors


# In[ ]:
n = 100  # number of reviewers
m = n  # number of proposals
k = 10  # number of proposals per reviewer
l = k  # number of reviewers per proposal
args = (n, m, k, l)
pr = 0.75  # probability of a reviwer getting the right ranking when comparing two
n_trials = 2
phi = 0.4
alpha = 0.5
scores = [np.exp(alpha * i) for i in range(m, 0, -1)]  # ground truth scores
gamma = 2
nabla = 0.001
pl_errors = np.zeros((2, n_trials))  # Row 0 is for disagreement errors, Row 2 is for KT to ground truth
borda_errors = np.zeros((2, n_trials))
kemeny_errors = np.zeros((2, n_trials))
position_errors = np.zeros((2, n_trials))
min_max_errors = np.zeros((2, n_trials))
rms_errors = np.zeros((2, n_trials))
g_weights = [1 for _ in range(m)]  # for computing KT to ground truth
g_max = m * (m - 1) / 2  # maximum possible non-weieghted KT
# In[]:
all_assignments = []
max_kts = []
while (len(all_assignments) < n_trials):
    print("Current length:", len(all_assignments))
    try:
        assignments = generate_matrix(args)
        all_assignments.append(assignments)
    except ValueError:
        print("skipped")

# In[]:

for z in tqdm(range(n_trials)):
    assignments = assignments = all_assignments[z]
    # reviews = plackett_luce(assignments, scores, args)
    reviews = mallows_generate(assignments, phi, args)

    reviewers, M, weights = reviewer_split(reviews, args, gamma)
    max_kt = 0  # The maximum weighted KT possible for this specific split
    for i in range(n):
        for j in range(i + 1, n):
            max_kt += weights[i] * weights[j]
    max_kts.append(max_kt)

    data1 = [reviews[j] for j in reviewers[0]]
    data2 = [reviews[j] for j in reviewers[1]]
    scores1 = choix.opt_rankings(m, data1)
    scores2 = choix.opt_rankings(m, data2)
    mle_error, mle_disagreements, mle_ties = kendall_tau_scores(weights, scores1, scores2, args)
    pl_errors[0, z] = mle_error / max_kt

    a_borda = torch.linspace(1.0, 0.0, steps=k)
    kt_borda, borda_disagreements, borda_ties = kendall_tau(M, weights, a_borda, args, nabla)
    borda_errors[0, z] = kt_borda / max_kt

    kemeny_ranking_1, score_1 = kemeny_gurobi_lazy(data1, args, printout_mode=False, time_out=60)
    kemeny_ranking_2, score_2 = kemeny_gurobi_lazy(data2, args, printout_mode=False, time_out=60)
    kemeny_scores_1 = np.argsort(np.flip(kemeny_ranking_1))
    kemeny_scores_2 = np.argsort(np.flip(kemeny_ranking_2))
    kemeny_error, kemeny_disagreements, kemeny_ties = kendall_tau_scores(weights, kemeny_scores_1, kemeny_scores_2,
                                                                         args)
    kemeny_errors[0, z] = kemeny_error / max_kt

    # positional_scores = position_scoring(M, weights, args)
    # kt_score_temp, disagreements, ties=kendall_tau(M, weights, positional_scores, args,nabla)
    # position_errors[0,z]=kt_score_temp/max_kt

    proposal_ranks1 = [[] for _ in range(m)]  # for each proposal, we will add its ranks here. (REVIEWERS 1)
    proposal_ranks2 = [[] for _ in range(m)]  # for each proposal, we will add its ranks here. (REVIEWERS 2)
    for f in range(n):
        if f in reviewers[0]:
            for j in range(len(reviews[f])):
                proposal_ranks1[reviews[f][j]].append(j)
        elif f in reviewers[1]:
            for j in range(len(reviews[f])):
                proposal_ranks2[reviews[f][j]].append(j)

    minmax_scores1 = ave_ranking_scores(args, proposal_ranks1, max_mix_rejection=True, rms_rejection=False)
    minmax_scores2 = ave_ranking_scores(args, proposal_ranks2, max_mix_rejection=True, rms_rejection=False)
    min_max_error, min_max_disagreements, min_max_ties = kendall_tau_scores(weights, minmax_scores1, minmax_scores2,
                                                                            args)
    min_max_errors[0, z] = min_max_error / max_kt

    rms_scores1 = ave_ranking_scores(args, proposal_ranks1, max_mix_rejection=False, rms_rejection=True)
    rms_scores2 = ave_ranking_scores(args, proposal_ranks2, max_mix_rejection=False, rms_rejection=True)
    rms_error, rms_disagreements, rms_ties = kendall_tau_scores(weights, rms_scores1, rms_scores2, args)
    rms_errors[0, z] = rms_error / max_kt

    # COMPUTING DISAGREEMENTS TO GROUND TRUTH:
    pl_mle_scores = choix.opt_rankings(m, reviews)
    pl_g_error, pl_g_disagreements, pl_g_ties = kendall_tau_scores(g_weights, scores, pl_mle_scores, args)
    pl_errors[1, z] = pl_g_error / g_max

    borda_scores = total_scores_all(M, a_borda, args)
    borda_g_error, borda_g_disagreements, borda_g_ties = kendall_tau_scores(g_weights, scores, borda_scores, args)
    borda_errors[1, z] = borda_g_error / g_max

    kemeny_output, score = kemeny_gurobi_lazy(reviews, args, printout_mode=False, time_out=60)
    kemeny_scores = np.argsort(np.flip(kemeny_output))
    kemeny_g_error, kemeny_g_disagreements, kemeny_g_ties = kendall_tau_scores(g_weights, scores, kemeny_scores, args)
    kemeny_errors[1, z] = kemeny_g_error / g_max

    # pos_scores=total_scores_all(M, positional_scores, args)
    # pos_g_error, pos_g_disagreements, pos_g_ties=kendall_tau_scores(g_weights, scores,pos_scores, args)
    # position_errors[1,z]=pos_g_error/g_max

    proposal_ranks_all = [[] for _ in range(m)]
    for f in range(m):
        proposal_ranks_all[f] = proposal_ranks1[f] + proposal_ranks2[f]

    minmax_scores = ave_ranking_scores(args, proposal_ranks_all, max_mix_rejection=True, rms_rejection=False)
    minmax_g_error, minmax_g_disagreements, minmax_g_ties = kendall_tau_scores(g_weights, scores, minmax_scores, args)
    min_max_errors[1, z] = minmax_g_error / g_max

    rms_scores = ave_ranking_scores(args, proposal_ranks_all, max_mix_rejection=False, rms_rejection=True)
    rms_g_error, rms_g_disagreements, rms_g_ties = kendall_tau_scores(g_weights, scores, rms_scores, args)
    rms_errors[1, z] = rms_g_error / g_max

# In[ ]:

p1, = plt.plot(pl_errors[1, :], pl_errors[0, :], 'r.', label='PL MLE')
p2, = plt.plot(borda_errors[1, :], borda_errors[0, :], 'b.', label='Borda')
p3, = plt.plot(kemeny_errors[1, :], kemeny_errors[0, :], 'g.', label='Kemeny')
# p4,=plt.plot(position_errors[1,:],position_errors[0,:],'c.',label='Best Scoring')
p5, = plt.plot(min_max_errors[1, :], min_max_errors[0, :], 'm.', label='Borda w/ Minmax')
p6, = plt.plot(rms_errors[1, :], rms_errors[0, :], 'k.', label='Borda w/ RMS')
plt.xlabel("(Normalized) KT distance to ground truth")
plt.ylabel("(Normalized) Disagreement error")
plt.title("Generator: PL, n=100, k=10")

# In[ ]:
from numpy.polynomial.polynomial import polyfit

all_data = np.concatenate((pl_errors, borda_errors, kemeny_errors, position_errors, min_max_errors, rms_errors), axis=1)
# all_data = np.concatenate((pl_errors, borda_errors, min_max_errors, rms_errors), axis=1)

b, m = polyfit(all_data[1, :], all_data[0, :], 1)

x = np.array([min(all_data[1, :]), max(all_data[1, :])])
bf, = plt.plot(x, b + m * x, '-', label=('Best fit: b=' + str(round(b, 3)) + ", m=" + str(round(m, 3))), color='orange')

# first_legend = plt.legend(handles=[p1,p2,p3,p4,p5,p6], loc='lower right')
first_legend = plt.legend(handles=[p1, p2, p5, p6], loc='lower right')

plt.gca().add_artist(first_legend)
plt.legend(handles=[bf], loc='upper left')

plt.show()

# In[ ]:


# borda_errors=np.load("borda_errors.npy")
# kemeny_errors=np.load("kemeny_errors.npy")
# min_max_errors=np.load("minmax_errors.npy")
# pl_errors=np.load("pl_errors.npy")
# position_errors=np.load("position_errors.npy")
# rms_errors=np.load("rms_errors.npy")


correlation_matrix = np.corrcoef(all_data[1, :], all_data[0, :])
