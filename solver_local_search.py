import copy
import numpy as np
import random


from solver_heuristic import solve_heuristic



def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    n_iter = 100

    puzzle = copy.deepcopy(eternity_puzzle)
    best_solution, best_n_conflict = solve_heuristic(puzzle)


    for _ in range(n_iter):

        puzzle = copy.deepcopy(eternity_puzzle)
        random.shuffle(puzzle.piece_list) 

        solution, n_conflict = solve_heuristic(puzzle)

        if n_conflict < best_n_conflict:
            best_solution = solution
            best_n_conflict = n_conflict

    return  best_solution, best_n_conflict