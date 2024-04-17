import copy
import numpy as np
import random
from copy import deepcopy
from itertools import combinations
from math import comb
from time import time
from os import path


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
def log_execution(instance, score, execution_time):
    # Format log message
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Write log message to file
    with open(path.join("log", "log_local_search.txt"), "a") as log_file:
        log_file.write(log_message)


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


def deux_swap_un_seul_valide(pieces, puzzle):
    """
    Génère une liste de voisins de la solution ...

    Args:
        pieces
        puzzle
    
    Returns:
        deux_swap_neigh (List[List])
    """

    valid = False
    corner_or_edge = False
    n = len(pieces)

    while not valid:
        # On tire deux pièces au hasard
        index_piece_1, index_piece_2 = np.random.choice(n, size=2, replace=False)
        piece1, piece2 = pieces[index_piece_1], pieces[index_piece_2]

        # On garde cette sélection si les deux pièces sont de la même géométrie (bords, coins)
        if np.count_nonzero(piece1) == np.count_nonzero(piece2):
            valid = True
            if np.count_nonzero(piece1) != 4:
                corner_or_edge = True

    neigh = deepcopy(pieces)

    if corner_or_edge:
        # On échange les deux bords/coins en échangeant leurs rotations
        corn_edge_piece_1 = np.where(np.array(pieces[index_piece_1]) == 0)[0]
        corn_edge_piece_2 = np.where(np.array(pieces[index_piece_2]) == 0)[0]

        #print(pieces[index_piece_1], corn_edge_piece_1)
        #print(pieces[index_piece_2], corn_edge_piece_2)

        if np.allclose(corn_edge_piece_1, corn_edge_piece_2):
            # Même orientation (bords du même côté du plateau)
            neigh[index_piece_1], neigh[index_piece_2] = pieces[index_piece_2], pieces[index_piece_1]

        else:
            if len(corn_edge_piece_1) == 1:
                # Les deux pièces sont des bords
                if corn_edge_piece_1[0] + corn_edge_piece_2[0] == 1 or corn_edge_piece_1[0] + corn_edge_piece_2[0] == 5:
                    # Les indices sont 0 et 1, ou 2 et 3 (bords opposés)
                    neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[2], puzzle.generate_rotation(pieces[index_piece_1])[2]

                elif corn_edge_piece_1[0] + corn_edge_piece_2[0] == 2:
                    # Bords haut et gauche
                    if corn_edge_piece_1[0] == 0:
                        # Pièce 1 est un bord haut : on doit la tourner de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]
                    else:
                        # Pièce 1 est un bord gauche : on doit la tourner de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]

                elif corn_edge_piece_1[0] + corn_edge_piece_2[0] == 4:
                    # Bords bas et droite
                    if corn_edge_piece_1[0] == 3:
                        # Pièce 1 est un bord droit : on doit la tourner de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]
                    else:
                        # Pièce 1 est un bord bas : on doit la tourner de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]
                
                elif abs(corn_edge_piece_1[0] + corn_edge_piece_2[0] == 4) == 3:
                    # Bords haut et droite
                    if corn_edge_piece_1[0] == 0:
                        # Pièce 1 est un bord haut : on doit la tourner de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]
                    else:
                        # Pièce 1 est un bord droit : on doit la tourner de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]
                
                else:
                    # Bords bas et gauche
                    if corn_edge_piece_1[0] == 1:
                        # Pièce 1 est un bord bas : on doit la touner de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]
                    else:
                        # Pièce 1 est un bord droit : on doit la tourner de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]
            
            else:
                # Les deux pièces sont des coins
                if np.sum(corn_edge_piece_1 == corn_edge_piece_2) == 0:
                    # Les deux pièces sont dans les coins opposés ; rotation de 180°
                    neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[2], puzzle.generate_rotation(pieces[index_piece_1])[2]

                elif corn_edge_piece_1[0] == corn_edge_piece_2[0]:
                    # np.where est séquentiel : les indices sont triés. 
                    # Premiers éléments égaux <=> les premiers éléments sont 0 (coins haut-G et haut-D) ou 1 (coins bas-G et bas-D)
                    if np.sum(corn_edge_piece_1) == 3:
                        # On doit tourner la pièce 1 de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]
                    else:
                        # On doit tourner la pièce 2 de 270°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]
                
                else:
                    # Premiers éléments diff <=> les deuxièmes éléments sont 2 (coins haut-G et bas-G) ou 3 (coins haut-D et bas-D)
                    if np.sum(corn_edge_piece_1) == 3:
                        # On doit tourner la pièce 1 de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[3], puzzle.generate_rotation(pieces[index_piece_1])[1]
                    else:
                        # On doit tourner la pièce 2 de 90°
                        neigh[index_piece_1], neigh[index_piece_2] = puzzle.generate_rotation(pieces[index_piece_2])[1], puzzle.generate_rotation(pieces[index_piece_1])[3]

        #print(neigh[index_piece_1], neigh[index_piece_2])
        #print()
    else:
        # Les deux pièces sont des pièces internes : on les échange en faisant éventuellement une rotation de l'une dans 25% des cas
        if np.random.rand() < 0.25:
            neigh[index_piece_1], neigh[index_piece_2] = pieces[index_piece_2], puzzle.generate_rotation(pieces[index_piece_1])[np.random.randint(0,4)]
        else:
            neigh[index_piece_1], neigh[index_piece_2] = pieces[index_piece_2], neigh[index_piece_1]


    return neigh


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

    time_credit = 3600

    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:
        # Temps de début du restart
        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        #restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
        temp = 10

        listeTabu = Tabu(20)

        for i in range(20000):
            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            solution = deux_swap_un_seul_valide(current_solution, puzzle)   

            if solution not in listeTabu:
                listeTabu.add(solution)
                n_conflict = puzzle.get_total_n_conflict(solution)

                delta = n_conflict - current_n_conflict

                if delta <= 0 or np.random.rand() < np.exp(-delta/temp):
                    current_solution, current_n_conflict = solution, n_conflict

                    # On choisit comme représentation courante le meilleur voisin

                if current_n_conflict < best_n_conflict_restart:
                    best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
                
                temp *= 0.99

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            best_solution = best_solution_restart
            print(best_n_conflict)
        
        iteration_duration = time() - t1

    log_execution(dict_board_size_instance[eternity_puzzle.board_size], best_n_conflict, time()-t0)

    return  best_solution, best_n_conflict
