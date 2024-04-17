import copy
import numpy as np
import random
from time import time
from collections import deque
from hungarian import HungarianAlg
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

NORD = 0
SUD = 1
OUEST = 2
EST = 3


class Tabu:
    def __init__(self, size: int):
        self.size = size
        self.tabuList = deque(maxlen=size)

    def add(self, element):
        self.tabuList.append(copy.deepcopy(element))

    def __contains__(self, element):
        return element in self.tabuList


def log_execution(instance, score, execution_time):


    # Format log message
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Write log message to file
    with open(path.join("log", "log_advanced.txt"), "a") as log_file:
        log_file.write(log_message)

def deux_swap(pieces, puzzle, board_size):
    """
    """
    n = len(pieces)
    couleurs_des_voisins = [couleurs_voisins(k, pieces, board_size) for k in range(n)]


    deux_swap_neigh = []
    pieces_gen_rotations = [puzzle.generate_rotation(pieces[i]) for i in range(n)]


    

    for i in range(n):
        for j in range(n):
            if j!=i:
                for k in range(4):
                    for m in range(4):
                        old_piece_i = pieces[i]
                        old_piece_j = pieces[j]
                        new_piece_i = pieces_gen_rotations[i][k]
                        new_piece_j = pieces_gen_rotations[j][m]
                        neigh, delta_swap = deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins)
                        deux_swap_neigh.append((delta_swap,neigh))
            else:
                for m in range(4):
                    old_piece_i = pieces[i]
                    old_piece_j = pieces[j]
                    new_piece_i = pieces[i]
                    new_piece_j = pieces_gen_rotations[j][m]
                    #print(deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins))
                    neigh, delta_swap = deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins)
                    deux_swap_neigh.append((delta_swap,neigh))

            
    return deux_swap_neigh

def perturbations(pieces,ratio):

    n = len(pieces)

    array_pieces=  np.array(pieces)

    indexes = np.random.choice(n, int(ratio*n), replace=False)

    elements = array_pieces[indexes].copy()

    np.random.shuffle(elements)

    
    array_pieces[indexes] = elements

    return [tuple(truc) for truc in array_pieces]

def deux_swap_with_worst_piece(pieces, puzzle, board_size):


    n = len(pieces)
    couleurs_des_voisins = [couleurs_voisins(k, pieces, board_size) for k in range(n)]
    conflict = [nb_conflits(pieces[i],couleurs_des_voisins[i]) for i in range(len(pieces))]

    deux_swap_neigh = []
    pieces_gen_rotations = [puzzle.generate_rotation(pieces[i]) for i in range(n)]


    index_worst = np.argmax(conflict)
    #print("nb_seul" ,conflict[index_worst])
    worst_piece, couleurs_des_voisins_worst = pieces[index_worst], couleurs_des_voisins[index_worst]


    for j in range(n):
        if j!=index_worst:
            for k in range(4):
                for m in range(4):
                    i = index_worst
                    old_piece_i = worst_piece
                    old_piece_j = pieces[j]
                    new_piece_i = pieces_gen_rotations[i][k]
                    new_piece_j = pieces_gen_rotations[j][m]
                    neigh, delta_swap, adj_ho, adj_ve, adj_rien = deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins)
                    deux_swap_neigh.append((delta_swap,neigh,i,j, adj_ho, adj_ve, adj_rien)) 
        
        else:
            for m in range(4):
                i = index_worst
                old_piece_i = worst_piece
                old_piece_j = worst_piece
                new_piece_i = worst_piece
                new_piece_j = pieces_gen_rotations[j][m]
                #print(deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins))
                neigh, delta_swap, adj_ho, adj_ve, adj_rien = deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins)
                deux_swap_neigh.append((delta_swap,neigh,i,j ,adj_ho, adj_ve, adj_rien))

    return deux_swap_neigh



def deux_swap_delta(i,j,old_piece_i, old_piece_j, new_piece_i, new_piece_j,pieces, board_size, couleurs_des_voisins):

    #On teste d'abord si les cases sont adjacentes et si oui, horizontalement ou verticalement
    indice_maximum = max(i,j)
    indice_minimum = min(i,j)

    couleurs_voisins_i = couleurs_des_voisins[i]
    couleurs_voisins_j = couleurs_des_voisins[j]


    correction_old = 0
    correction_new = 0

    neigh = pieces.copy()

    adj_ho = False
    adj_ve = False
    adj_rien = False

    #La même pièce
    if i == j:
        
        old_conflict = nb_conflits(old_piece_i, couleurs_voisins_i)
        new_conflict = nb_conflits(new_piece_j, couleurs_voisins_i)

        neigh[i] = new_piece_j
        delta_swap= new_conflict - old_conflict

        return neigh, delta_swap

    #adjacence horizontale
    elif (indice_maximum-indice_minimum) == 1 and (indice_maximum % board_size != 0 ):
        adj_ho = True
        if pieces[indice_maximum][OUEST]!=pieces[indice_minimum][EST]:
            correction_old += 1
        
        neigh[i], neigh[j] = new_piece_j, new_piece_i
        if neigh[indice_maximum][OUEST]!=neigh[indice_minimum][EST]:
            correction_new += 1

        couleurs_voisins_i_adjacent = couleurs_voisins(i, neigh, board_size)
        couleurs_voisins_j_adjacent = couleurs_voisins(j, neigh, board_size)


    #adjacence verticale
    elif indice_maximum-indice_minimum == board_size :
        adj_ve = True
        if pieces[indice_maximum][SUD]!=pieces[indice_minimum][NORD]:
            correction_old += 1
        neigh[i], neigh[j] = new_piece_j, new_piece_i
        if neigh[indice_maximum][SUD]!=neigh[indice_minimum][NORD]:
            correction_new += 1

        couleurs_voisins_i_adjacent = couleurs_voisins(i, neigh, board_size)
        couleurs_voisins_j_adjacent = couleurs_voisins(j, neigh, board_size)

    # Pas d'adjacence
    else:
        adj_rien = True
        neigh[i], neigh[j] = new_piece_j, new_piece_i
        couleurs_voisins_i_adjacent = couleurs_voisins_i
        couleurs_voisins_j_adjacent = couleurs_voisins_j

    return neigh, (calcul_cout_swap(new_piece_j, new_piece_i, couleurs_voisins_i_adjacent, couleurs_voisins_j_adjacent)-correction_new - calcul_cout_swap(old_piece_i, old_piece_j, couleurs_voisins_i, couleurs_voisins_j) + correction_old)


    


def couleurs_voisins(i : int,pieces : list[tuple],board_size : int):
    """
    Args:
        i  (int) : position de la pièce dans la matrice
        pieces (List[tuple]) : matrice des pièces du puzzle
        board_size (int) : taille du plateau
    
    """


    #on récupère les indices des voisins de la pièce
    k_est_1 = i + 1
    k_ouest_1 = i - 1
    k_sud_1 = i - board_size
    k_nord_1 = i + board_size


    #On récupère les couleurs des voisins
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
    """
    Give the total number of conflicts of two pieces in the matrice
    Args:
        piece_1 (Tuple[int])
        piece_2 (Tuple[int])
        couleurs_voisins_1 (Tuple[int])
        couleurs_voisins_2 (Tuple[int])
    Returns:
        cout (Int) : total number of conflictsS

    """
    
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

    time_credit = 300
    board_size = puzzle.board_size
    


    listeTabu = Tabu(5)
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

        listeTabu = Tabu(5)

        for i in range(100):
            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            solution = sorted(deux_swap(current_solution, puzzle, board_size),key= lambda x : x[0])[0][1]

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