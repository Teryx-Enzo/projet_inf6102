import copy
import numpy as np
import random
from time import time
from collections import deque
from solver_heuristic import solve_heuristic
from tqdm import tqdm
from os import path

GRIS = 0

random.seed(2306632)

dict_board_size_instance = {
    4 : 'A',
    7 : 'B',
    8 : 'C',
    9 : 'D',
    10 : 'E',
    16 : 'complet'
}
def log_execution(instance, score, execution_time):
    # Format log message
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Write log message to file
    with open(path.join("log", "log_local_search.txt"), "a") as log_file:
        log_file.write(log_message)

NORD = 0
SUD = 1
OUEST = 2
EST = 3


def destroy(solution, degree: float):
    """
    Destruction d'une solution

    Args:
        solution
        degree (float) : poucentage de pièces à retirer

    Returns:
    """
    destroyed_sol = solution.copy()

    n_tiles = np.product(solution.shape[:2])

    indexes = np.unravel_index(np.random.choice(n_tiles, size = int(degree*n_tiles), replace=False),
                                                solution.shape[:2])

    removed_elements = destroyed_sol[indexes]
    destroyed_sol[indexes] = -1

    return destroyed_sol, removed_elements


def greedy_replace(solution, tile, coords):
    """
    Remet la tuile tile dans la solution aux coordonnées coordinates, dans l'orientation minimisant le mieux les conflits.
    """
    board_size = solution.shape[0]

    initial_shape = tile
    rotation_90 = (tile[2], tile[3], tile[1], tile[0])
    rotation_180 = (tile[1], tile[0], tile[3], tile[2])
    rotation_270 = (tile[3], tile[2], tile[0], tile[1])

    orientations = [initial_shape, rotation_90, rotation_180, rotation_270]

    conflicts = []
    for orient in orientations:
        conflicts.append(0)

        if coords[0] == 0:
            # Ligne du haut : on veut que l'élément haut soit gris
            conflicts[-1] += 2*int(orient[0] != 0)
        else:
            conflicts[-1] += int(orient[0] != solution[coords[0]-1, coords[1], 1])

        if coords[0] == board_size-1:
            conflicts[-1] += 2*int(orient[1] != 0)
        else:
            conflicts[-1] += int(orient[1] != solution[coords[0]+1, coords[1], 0])
        
        if coords[1] == 0:
            conflicts[-1] += 2*int(orient[2] != 0)
        else:
            conflicts[-1] += int(orient[2] != solution[coords[0], coords[1]-1, 3])

        if coords[1] == board_size-1:
            conflicts[-1] += 2*int(orient[3] != 0)
        else:
            conflicts[-1] += int(orient[3] != solution[coords[0], coords[1]+1, 2])
    
    return orientations[np.argmin(conflicts)]


def reconstruct(solution, removed_elements):
    """
    D

    Args:

    Returns:
    """
    board_size = solution.shape[0]

    reconstructed_sol = solution.copy()
    to_reassign = np.where(reconstructed_sol[:, :, 0] == -1)
    nonzero_per_removed_tile = np.count_nonzero(removed_elements, axis=1)

    # print(solution)
    # print(to_reassign)
    # print(nonzero_per_removed_tile)

    for e in np.transpose(to_reassign):
        # print(e, nonzero_per_removed_tile)
        if (np.allclose(e, np.array([0, 0])) or
            np.allclose(e, np.array([0, board_size-1])) or
            np.allclose(e, np.array([board_size-1, 0])) or
            np.allclose(e, np.array([board_size-1, board_size-1]))):
            # Dans un coin
            # print('coin')
            i = np.random.choice(np.where(nonzero_per_removed_tile == 2)[0])

        elif (e[0] == 0 or 
              e[0] == board_size-1 or
              e[1] == 0 or
              e[1] == board_size-1):
            # Sur un bord
            # print('bord')
            i = np.random.choice(np.where(nonzero_per_removed_tile == 3)[0])

        else:
            # milieu
            # print('milieu')
            i = np.random.choice(np.where(nonzero_per_removed_tile == 4)[0])

        reconstructed_sol[e[0], e[1]] = greedy_replace(reconstructed_sol, removed_elements[i], e) 

        nonzero_per_removed_tile = np.delete(nonzero_per_removed_tile, i, 0)
        removed_elements = np.delete(removed_elements, i, 0)

    return reconstructed_sol



def evaluate(puzzle, solution):
    """
    E
    """
    return puzzle.get_total_n_conflict(
        np.reshape(np.flip(solution, axis=0), newshape=(-1, 4))
    )


def solve_advanced(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    n = eternity_puzzle.board_size

    puzzle = copy.deepcopy(eternity_puzzle)
    best_solution, best_n_conflict = solve_heuristic(puzzle)

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    time_credit = 300

    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:
        # Temps de début du restart
        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        #restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        current_solution = np.flip(np.array(current_solution).reshape((n, n, 4)), 0)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict

        temp = 10
     
        for i in range(500):
            solution = reconstruct(*destroy(current_solution, 0.2))

            # Acceptation éventuelle de la solution reconstruite
            n_conflict = evaluate(puzzle, solution)
            delta = n_conflict - current_n_conflict

            if delta <= 0:# or np.random.rand() < np.exp(-delta/temp):
                current_solution, current_n_conflict = solution, n_conflict

            if current_n_conflict < best_n_conflict_restart:
                best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
            
            temp *= 0.99

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            best_solution = best_solution_restart
            print(best_n_conflict)
        
        iteration_duration = time() - t1

    #log_execution(dict_board_size_instance[eternity_puzzle.board_size], best_n_conflict, time()-t0)

    # Mise en forme pour la visualisation
    best_solution = [
        tuple(e) for e in np.reshape(np.flip(best_solution, axis=0), newshape=(-1, 4))
    ]

    return  best_solution, best_n_conflict