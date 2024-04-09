import copy
import numpy as np
import random
from itertools import combinations
from math import comb
from time import time

# Importation du solveur heuristique
from solver_heuristic import solve_heuristic

# Constante pour la couleur GRIS
GRIS = 0

# Initialisation de la seed pour la reproductibilité
random.seed(2306632)

# Dictionnaire pour mapper les tailles de plateau aux lettres
dict_board_size_instance = {
    4 : 'A',
    7 : 'B',
    8 : 'C',
    9 : 'D',
    10 : 'E',
    16 : 'complet'
}

# Constantes pour les directions
NORD = 0
SUD = 1
OUEST = 2
EST = 3

def log_execution(instance, score, execution_time):
    """
    Fonction pour enregistrer les résultats de l'exécution dans un fichier de log.

    Args:
        instance (str): Instance du problème.
        score (int): Score de la solution.
        execution_time (float): Temps d'exécution.
    """
    # Format du message de log
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Écriture du message de log dans le fichier
    with open("log_advanced.txt", "a") as log_file:
        log_file.write(log_message)

def deux_swap(pieces, puzzle, board_size):
    """
    Génère une liste de voisins de la solution en échangeant pour chaque voisin 2 tableaux distincts.

    Args:
        pieces (List): Liste d'items de dictionnaire (index, ArtPiece), solution actuelle.
        puzzle (object): Instance de l'objet EternityPuzzle.
        board_size (int): Taille du plateau.

    Returns:
        deux_swap_neigh (List[List]): Liste de voisins de pieces.
    """
    n = len(pieces)
    deux_swap_neigh = []
    # Précalculer les rotations pour chaque pièce
    rotated_pieces = [puzzle.generate_rotation(piece) for piece in pieces]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(4):
                for m in range(4):
                    # Utiliser des références aux pièces originales pour éviter la copie profonde inutile
                    piece1 = rotated_pieces[i][k]
                    piece2 = rotated_pieces[j][m]
                    # Créer une nouvelle liste de pièces en échangeant les pièces sélectionnées
                    neigh = pieces[:]
                    neigh[i], neigh[j] = piece2, piece1
                    # Calculer le coût du swap en utilisant directement les pièces originales
                    cout_swap = calcul_cout_swap(piece2, pieces[j], i, j, pieces, board_size)
                    deux_swap_neigh.append((cout_swap, neigh))
    return deux_swap_neigh

def couleurs_voisins(i, pieces, board_size):
    """
    Calcule les couleurs des voisins pour une pièce donnée.

    Args:
        i (int): Indice de la pièce.
        pieces (List): Liste d'items de dictionnaire (index, ArtPiece).
        board_size (int): Taille du plateau.

    Returns:
        tuple: Tuple contenant les couleurs des voisins (nord, sud, ouest, est).
    """
    k_est_1 = i + 1
    k_ouest_1 = i - 1
    k_sud_1 = i - board_size
    k_nord_1 = i + board_size

    if i < board_size:
        c_sud_1 = GRIS
    else:
        c_sud_1 = pieces[k_sud_1][NORD]
    
    if i % board_size == board_size - 1:
        c_est_1 = GRIS
    else:
        c_est_1 = pieces[k_est_1][OUEST]

    if i > board_size**2 - board_size - 1:
        c_nord_1 = GRIS
    else:
        c_nord_1 = pieces[k_nord_1][SUD]

    if i % board_size == 0:
        c_ouest_1 = GRIS
    else:
        c_ouest_1 = pieces[k_ouest_1][EST]

    return (c_nord_1 , c_sud_1, c_ouest_1, c_est_1)

def calcul_cout_swap(piece_1, piece_2, i, j, pieces, board_size):
    """
    Calcule le coût de swap entre deux pièces.

    Args:
        piece_1 (tuple): Pièce 1.
        piece_2 (tuple): Pièce 2.
        i (int): Indice de la pièce 1.
        j (int): Indice de la pièce 2.
        pieces (List): Liste d'items de dictionnaire (index, ArtPiece).
        board_size (int): Taille du plateau.

    Returns:
        int: Coût du swap.
    """
    couleurs_voisins_1 = couleurs_voisins(i, pieces, board_size)
    couleurs_voisins_2 = couleurs_voisins(j, pieces, board_size)
    return points_communs(piece_1, couleurs_voisins_2) + points_communs(piece_2, couleurs_voisins_1)

def points_communs(tuple1, tuple2):
    """
    Calcule le nombre de points communs entre deux tuples.

    Args:
        tuple1 (tuple): Premier tuple.
        tuple2 (tuple): Deuxième tuple.

    Returns:
        int: Nombre de points communs.
    """
    score = 0
    for i in range(len(tuple1)):
        if tuple1[i] == tuple2[i] == 0:
            score += 2
        elif tuple1[i] == tuple2[i]:
            score += 1
        elif tuple1[i] == 0:
            score -= 1
    return score

def solve_advanced(eternity_puzzle):
    """
    Solution de recherche locale du problème.

    Args:
        eternity_puzzle (object): Objet décrivant l'entrée.

    Returns:
        tuple: Tuple contenant la meilleure solution et le coût de la solution.
    """
    n_iter = 10

    puzzle = copy.deepcopy(eternity_puzzle)
    best_solution, best_n_conflict = solve_heuristic(puzzle)

    # Métriques de temps d'exécution
    t0 = time()
    iteration_duration = 0
    time_credit = 360
    board_size = puzzle.board_size

    # Recherche
    while ((time() - t0) + iteration_duration) < time_credit - 5:
        t1 = time()
        puzzle = copy.deepcopy(eternity_puzzle)
        # Restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict
        temp = 3

        for _ in range(20):
            # On choisit un voisin au hasard, et on le garde s'il est améliorant ou selon une probabilité dépendant de son coût et de la température.
            solution = sorted(deux_swap(current_solution, puzzle, board_size), key=lambda k: k[0])[0][1]
            n_conflict = puzzle.get_total_n_conflict(solution)
            delta = n_conflict - current_n_conflict

            if delta <= 0 or np.random.rand() < np.exp(-delta/temp):
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

    # Enregistrement des résultats
    log_execution(dict_board_size_instance[eternity_puzzle.board_size], best_n_conflict, time() - t0)
    return best_solution, best_n_conflict
