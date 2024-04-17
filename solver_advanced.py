import copy
import numpy as np
import random
from time import time
from collections import deque
from hungarian import HungarianAlg
from solver_heuristic import solve_heuristic
from tqdm import tqdm

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
    with open("log_advanced.txt", "a") as log_file:
        log_file.write(log_message)

def deux_swap(pieces, puzzle, board_size):
    """
    """
    couleurs_des_voisins = [couleurs_voisins(k, pieces, board_size) for k in range(len(pieces))]
    n = len(pieces)
    deux_swap_neigh = []
    pieces_gen_rotations = [puzzle.generate_rotation(pieces[i]) for i in range(n)]


    for i in range(n):
        for j in range(i, n):
            couleurs_voisins_1 = couleurs_des_voisins[i]
            couleurs_voisins_2 = couleurs_des_voisins[j]
            piece1_old = pieces[i]
            piece2_old = pieces[j]
            old_conflict = calcul_cout_swap(piece1_old, piece2_old, couleurs_voisins_1, couleurs_voisins_2)
            for k in range(4):
                for m in range(4):
                    neigh = pieces.copy() 
                    piece1 = pieces_gen_rotations[i][k]
                    piece2 = pieces_gen_rotations[j][m]
                    neigh[i], neigh[j] = piece2, piece1
                    cout_swap = calcul_cout_swap(piece2, piece1, couleurs_voisins_1, couleurs_voisins_2) - old_conflict
                    deux_swap_neigh.append((cout_swap,neigh))
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

    for i in range(n):
        
            couleurs_voisins_1 = couleurs_des_voisins[i]
            couleurs_voisins_2 = couleurs_des_voisins_worst
            piece1_old = pieces[i]
            piece2_old = worst_piece
            old_conflict = calcul_cout_swap(piece1_old, piece2_old, couleurs_voisins_1, couleurs_voisins_2)
            #print("nb_avant" ,(old_conflict))
            for k in range(4):
                for m in range(4):
                    neigh = pieces.copy() 
                    piece1 = pieces_gen_rotations[i][k]
                    piece2 = pieces_gen_rotations[index_worst][m]
                    
                    neigh[i], neigh[index_worst] = piece2, piece1
                    cout_swap = calcul_cout_swap(piece2, piece1, couleurs_voisins_1, couleurs_voisins_2) - old_conflict
                    print(piece1_old, piece2_old,piece1,piece2,couleurs_voisins_1, couleurs_voisins_2, cout_swap)
                    #print("nouveau_cout", cout_swap)
                    deux_swap_neigh.append((cout_swap,neigh))
    return deux_swap_neigh

def deux_swap_random(pieces, puzzle, board_size):
    print(pieces)

    n = len(pieces)
    couleurs_des_voisins = [couleurs_voisins(k, pieces, board_size) for k in range(n)]
    conflict = [nb_conflits(pieces[i],couleurs_des_voisins[i]) for i in range(len(pieces))]

    deux_swap_neigh = []
    pieces_gen_rotations = [puzzle.generate_rotation(pieces[i]) for i in range(n)]


    index_worst = np.argmax(conflict)
    #print("nb_seul" ,conflict[index_worst])
    worst_piece, couleurs_des_voisins_worst = pieces[index_worst], couleurs_des_voisins[index_worst]

    i = np.random.choice(n)
    while i == index_worst:
        i = np.random.choice(n)
        
    couleurs_voisins_1 = couleurs_des_voisins[i]
    couleurs_voisins_2 = couleurs_des_voisins_worst
    piece1_old = pieces[i]
    piece2_old = worst_piece
    old_conflict = calcul_cout_swap(piece1_old, piece2_old, couleurs_voisins_1, couleurs_voisins_2)
    #print("nb_avant" ,(old_conflict))
    for k in range(4):
        m = 0
        neigh = pieces.copy() 
        piece1 = pieces_gen_rotations[i][k]
        piece2 = pieces_gen_rotations[index_worst][m]
        
        neigh[i], neigh[index_worst] = piece2, piece1
        cout_swap = calcul_cout_swap(piece2, piece1, couleurs_voisins_1, couleurs_voisins_2) - old_conflict
        print(piece1_old, piece2_old,piece1,piece2,couleurs_voisins_1, couleurs_voisins_2, cout_swap)
            #print("nouveau_cout", cout_swap)
        deux_swap_neigh.append((cout_swap,neigh))
    return deux_swap_neigh
    

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

    time_credit = 360
    board_size = puzzle.board_size
    


    listeTabu = Tabu(5)
    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:

        

        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        #restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        #current_solution, current_n_conflict = solve_heuristic(puzzle)
        current_solution, current_n_conflict = puzzle.piece_list, eternity_puzzle.get_total_n_conflict(puzzle.piece_list)
        listeTabu.add(current_solution)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
        temp = 5

        
        for i in tqdm(range(5000)):


            
            if True:
                # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.

                #solution_possibles = sorted(deux_swap(current_solution, puzzle,board_size), key=lambda k: k[0])
                #solution_possibles = sorted(deux_swap_with_worst_piece(current_solution, puzzle,board_size), key=lambda k: k[0])
                solution_possibles = sorted(deux_swap_random(current_solution, puzzle,board_size), key=lambda k: k[0])


                
                
                selected = False
                i = 0
                while not selected :
                    
                    solution = solution_possibles[i][1]
                    
                    if not solution in listeTabu:
                        print("cout_retenu",solution_possibles[i][0] )
                        selected = True
                        listeTabu.add(solution)
                        
                        n_conflict = puzzle.get_total_n_conflict(solution)

                        

                        delta = n_conflict - current_n_conflict
                        print(delta)

                        if delta <= 0 or np.random.rand() < np.exp(-delta/temp):
                            # On choisit comme représentation courante le meilleur voisin
                            current_solution, current_n_conflict = solution, n_conflict


                    else : 

                        i+=1
            
                

            if current_n_conflict <= best_n_conflict_restart:
                
                best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict

            
            temp *= 0.98

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            best_solution = best_solution_restart
        



        print(best_n_conflict, best_n_conflict_restart, current_n_conflict)
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