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

NORD = 0
SUD = 1
OUEST = 2
EST = 3


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
    with open("log_advanced.txt", "a") as log_file:
        log_file.write(log_message)

def deux_swap(pieces,puzzle,board_size):
    """
    Génère une liste de voisins de la solution artpieces en échangeant pour chaque voisin 2 tableaux distincts.

    Args:
        artpieces (List): Liste d'items de dictionnaire (index, ArtPiece), solution actuelle.
    
    Returns:
        deux_swap_neigh (List[List]) : Liste de voisins de artpieces.
    """

    

    n = len(pieces)
    deux_swap_neigh = []
    for i in range(n):
        for j in range(i+1,n):
            couleurs_voisins_1 = couleurs_voisins(i,pieces,board_size)
            couleurs_voisins_2 = couleurs_voisins(j,pieces,board_size)
            piece1_old = pieces[i]
            piece2_old = pieces[j]
            old_conflict = calcul_cout_swap(piece1_old,piece2_old,couleurs_voisins_1,couleurs_voisins_2)

            #print(couleurs_voisins_1,couleurs_voisins_2,piece1_old,piece2_old,old_conflict)
            for k in range(4):
                for m in range(4):
                    neigh = deepcopy(pieces)
                    piece1 = puzzle.generate_rotation(pieces[i])[k]
                    piece2 = puzzle.generate_rotation(pieces[j])[m]

                    neigh[i], neigh[j] = piece2, piece1
                    cout_swap = calcul_cout_swap(piece2,piece1,couleurs_voisins_1,couleurs_voisins_2) - old_conflict
                    deux_swap_neigh.append((cout_swap,neigh))

    
    

    return deux_swap_neigh


def couleurs_voisins(i,pieces,board_size):



    k_est_1 = i + 1
    k_ouest_1 = i - 1
    k_sud_1 = i - board_size
    k_nord_1 = i + board_size


    if i < board_size:
        c_sud_1 = GRIS

    else : 
        c_sud_1 = pieces[k_sud_1][NORD]
    
    if i% board_size == board_size-1:
        c_est_1 = GRIS
    else :
        c_est_1 = pieces[k_est_1][OUEST]

    if i > board_size**2-board_size-1:
        c_nord_1 = GRIS
    else :
        c_nord_1 = pieces[k_nord_1][SUD]

    if i% board_size == 0:
    
        c_ouest_1 = GRIS

    else:
        c_ouest_1 = pieces[k_ouest_1][EST]

    

    return (c_nord_1 ,c_sud_1,c_ouest_1,c_est_1)

def calcul_cout_swap(piece_1,piece_2,couleurs_voisins_1,couleurs_voisins_2):


    cout = nb_conflits(piece_1,couleurs_voisins_1) +nb_conflits(piece_2,couleurs_voisins_2)

    
    return cout


def nb_conflits(tuple1,tuple2):


    nb_conflit = 0

    for i in range(len(tuple1)):
        if tuple1[i] != tuple2[i]:
            nb_conflit += 1



    return nb_conflit


def solve_advanced(eternity_puzzle):
    """
    Local search solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    n_iter = 10

    puzzle = copy.deepcopy(eternity_puzzle)
    best_solution, best_n_conflict = solve_heuristic(puzzle)
    

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0

    time_credit = 1200
    board_size = puzzle.board_size

    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:

        listeTabu = Tabu(30)

        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        #restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        listeTabu.add(current_solution)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
        temp = 3

        for i in range(50):

            
        

            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            solution_possibles = sorted(deux_swap(current_solution, puzzle,board_size), key=lambda k: k[0])
            
            
            selected = False
            i = 0
            while not selected :
                solution = solution_possibles[i][1]

                if not solution in listeTabu:
                    selected = True
                    listeTabu.add(solution)

                    n_conflict = puzzle.get_total_n_conflict(solution)

                    print(solution_possibles[i][0], n_conflict, current_n_conflict)

                    delta = n_conflict - current_n_conflict

                    if delta <= 0 or np.random.rand() < np.exp(-delta/temp):
                        current_solution, current_n_conflict = solution, n_conflict


                        # On choisit comme représentation courante le meilleur voisin
                        current_solution = solution
                else : i+=1
            
                

            if current_n_conflict <= best_n_conflict_restart:
                print(current_n_conflict)
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