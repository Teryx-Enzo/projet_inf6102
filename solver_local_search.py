import copy
import numpy as np
import random
from copy import deepcopy
from itertools import combinations
from math import comb
from time import time


from solver_heuristic import solve_heuristic

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

class Tabu:
    def __init__(self, size: int):
        self.size = size
        self.tabuList = []

    def add(self, element):
        if len(self.tabuList) < self.size:
            self.tabuList.append(copy.deepcopy(element))
        else:
            self.tabuList.pop(0)
            self.tabuList.append(copy.deepcopy(element))

    def __contains__(self, element):
        return element in self.tabuList

def log_execution(instance, score, execution_time):


    # Format log message
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Write log message to file
    with open("log_local_search.txt", "a") as log_file:
        log_file.write(log_message)

def deux_swap_un_seul_valide(pieces,puzzle):
    """
    Génère une liste de voisins de la solution artpieces en échangeant pour chaque voisin 2 tableaux distincts.

    Args:
        artpieces (List): Liste d'items de dictionnaire (index, ArtPiece), solution actuelle.
    
    Returns:
        deux_swap_neigh (List[List]) : Liste de voisins de artpieces.
    """

    valid = False
    bord = False
    n = len(pieces)


    index_piece_1 = np.random.randint(0,n)
    piece1 = pieces[index_piece_1]

    while not valid:
        index_piece_2 = np.random.randint(0,n)
        piece2 = pieces[index_piece_2]
        if np.count_nonzero(piece1) == np.count_nonzero(piece2):
            valid = True
            if np.count_nonzero(piece1) != 4:
                bord = True

    neigh = deepcopy(pieces)
    if bord :
        neigh[index_piece_1], neigh[index_piece_2] = pieces[index_piece_2], pieces[index_piece_1]
    else :
        neigh[index_piece_1], neigh[index_piece_2] = pieces[index_piece_2], puzzle.generate_rotation(pieces[index_piece_1])[np.random.randint(0,4)]


    return neigh
    

    # n = len(pieces)
    # deux_swap_neigh = []
    # for i in range(n):
    #     for j in range(i+1,n):
    #         neigh = deepcopy(pieces)
    #         neigh[i], neigh[j] = pieces[j], pieces[i]
    #         deux_swap_neigh.append(neigh)

    # arr = np.tile(pieces, (comb(len(pieces), 2), 1))
    # indices = np.array(list(combinations(range(len(pieces)), 2)))
    # print(indices,np.flip(indices, axis=-1))
    # arr[np.arange(arr.shape[0])[:, None], indices] = arr[np.arange(arr.shape[0])[:, None], np.flip(indices, axis=-1)]

    

    return deux_swap_neigh

def solve_local_search(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """

    puzzle = copy.deepcopy(eternity_puzzle)
    best_solution, best_n_conflict = solve_heuristic(puzzle)
    

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    time_credit = 120


    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:

        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        #restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
        temp = 3

        listeTabu = Tabu(300)

        for _ in range(20000):

            
            

            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            solution = deux_swap_un_seul_valide(current_solution, puzzle)   

            if solution not in listeTabu:
                listeTabu.add(solution)
                n_conflict = puzzle.get_total_n_conflict(solution)

                #print(n_conflict)  

                #print(n_conflict)
                delta = n_conflict - current_n_conflict

                if delta <= 0 :#or np.random.rand() < np.exp(-delta/temp):
                    current_solution, current_n_conflict = solution, n_conflict

                    # On choisit comme représentation courante le meilleur voisin
                    current_solution = solution
                
                    

                if current_n_conflict <= best_n_conflict_restart:
                    best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
                
                temp *= 0.98

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            best_solution = best_solution_restart
        
        iteration_duration = time() - t1


    log_execution(dict_board_size_instance[eternity_puzzle.board_size],best_n_conflict,time()-t0)
    return  best_solution, best_n_conflict

    for _ in range(n_iter):

        puzzle = copy.deepcopy(eternity_puzzle)
        random.shuffle(puzzle.piece_list) 

        solution, n_conflict = solve_heuristic(puzzle)



        if n_conflict < best_n_conflict:
            best_solution = solution
            best_n_conflict = n_conflict

    return  best_solution, best_n_conflict