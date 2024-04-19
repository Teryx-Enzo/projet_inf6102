import copy
import numpy as np
import random
from time import time
from solver_heuristic import solve_heuristic
from os import path


# LOG
dict_board_size_instance = {
    4 : 'A',
    7 : 'B',
    8 : 'C',
    9 : 'D',
    10 : 'E',
    16 : 'complet'
}
def log_execution(instance, score, execution_time):
    """
    Ajoute dans le fichier de log du solveur les performances obtenues.

    Args:
        instance (str) : le nom de l'instance résolue
        score (int) : le nombre de conflits
        execution_time (float) : le temps pris (pour base de comparaison)
    """
    # Format log message
    log_message = f"Instance: {instance}, Score: {score}, Execution Time: {execution_time:.4f} seconds\n"
    
    # Write log message to file
    with open(path.join("log", "log_advanced.txt"), "a") as log_file:
        log_file.write(log_message)


# RÉSOLUTION
def conflicts_for_tile(solution, tile, coords):
    """
    Calcule le nombre de conflits pour la tuile (avec son orientation) à la position coords dans la solution.

    Args:
        solution (np.ndarray) : la représentation 2D de la solution
        tile (np.ndarray or Tuple[int]) : la tuile avec son orientation
        coords (np.ndarray or List[int]) : les coordonnées dans la solution

    Returns:
        conf (int) : le nombre de conflits pour la tuile tile aux coordonnées coords dans l'orientation donnée
    """
    board_size = solution.shape[0]

    conf = 0

    if coords[0] == 0:
        # Ligne du haut : on veut que l'élément haut soit gris
        conf += int(tile[0] != 0)
    else:
        conf += int(tile[0] != solution[coords[0]-1, coords[1], 1] if solution[coords[0]-1, coords[1], 1] != -1 else 0)

    if coords[0] == board_size-1:
        conf += int(tile[1] != 0)
    else:
        conf += int(tile[1] != solution[coords[0]+1, coords[1], 0] if solution[coords[0]+1, coords[1], 0] != -1 else 0)
    
    if coords[1] == 0:
        conf += int(tile[2] != 0)
    else:
        conf += int(tile[2] != solution[coords[0], coords[1]-1, 3] if solution[coords[0], coords[1]-1, 3] != -1 else 0)

    if coords[1] == board_size-1:
        conf += int(tile[3] != 0)
    else:
        conf += int(tile[3] != solution[coords[0], coords[1]+1, 2] if solution[coords[0], coords[1]+1, 2] != -1 else 0)
    
    return conf


def destroy(solution, degree: float):
    """
    Destruction d'une solution en retirant degree% de pièces choisies aléatoirement.

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


def destroy_worst_tiles(solution, degree: float):
    """
    Destruction d'une solution en retirant degree% de pièces choisies selon leur nombre de conflits.

    Args:
        solution
        degree (float) : poucentage de pièces à retirer

    Returns:
    """
    destroyed_sol = solution.copy()
    
    n_tiles = np.product(solution.shape[:2])

    p = np.array([1 + conflicts_for_tile(solution, solution[np.unravel_index(n, solution.shape[:2]) ], np.unravel_index(n, solution.shape[:2]) )
                  for n in range(n_tiles)])
    p = p/p.sum()

    indexes = np.unravel_index(np.random.choice(n_tiles, size = int(degree*n_tiles), replace=False, p=p),
                                                solution.shape[:2])

    removed_elements = destroyed_sol[indexes]
    destroyed_sol[indexes] = -1

    return destroyed_sol, removed_elements


def greedy_replace(solution, tile, coords):
    """
    Remet la tuile tile dans la solution aux coordonnées coordinates, dans l'orientation minimisant le mieux les conflits.
    """
    initial_shape = tile
    rotation_90 = (tile[2], tile[3], tile[1], tile[0])
    rotation_180 = (tile[1], tile[0], tile[3], tile[2])
    rotation_270 = (tile[3], tile[2], tile[0], tile[1])

    orientations = [initial_shape, rotation_90, rotation_180, rotation_270]

    conflicts = [conflicts_for_tile(solution, orient, coords) for orient in orientations]
    
    return orientations[np.argmin(conflicts)]


def reconstruct_randomized_greedy(solution, removed_elements,puzzle):
    """
    D

    Args:

    Returns:
    """
    board_size = solution.shape[0]

    reconstructed_sol = solution.copy()
    to_reassign = np.where(reconstructed_sol[:, :, 0] == -1)
    nonzero_per_removed_tile = np.count_nonzero(removed_elements, axis=1)

    for e in np.transpose(to_reassign):
        if (np.allclose(e, np.array([0, 0])) or
            np.allclose(e, np.array([0, board_size-1])) or
            np.allclose(e, np.array([board_size-1, 0])) or
            np.allclose(e, np.array([board_size-1, board_size-1]))):
            # Dans un coin
            i = np.random.choice(np.where(nonzero_per_removed_tile == 2)[0])

        elif (e[0] == 0 or 
              e[0] == board_size-1 or
              e[1] == 0 or
              e[1] == board_size-1):
            # Sur un bord
            i = np.random.choice(np.where(nonzero_per_removed_tile == 3)[0])

        else:
            # milieu
            i = np.random.choice(np.where(nonzero_per_removed_tile == 4)[0])

        reconstructed_sol[e[0], e[1]] = greedy_replace(reconstructed_sol, removed_elements[i], e) 

        nonzero_per_removed_tile = np.delete(nonzero_per_removed_tile, i, 0)
        removed_elements = np.delete(removed_elements, i, 0)

    return reconstructed_sol



def reconstruct_deterministic_greedy(solution, removed_elements,puzzle, return_pos = False):
    """
    D

    Args:
        solution (np.ndarray) : l'encodage de la solution (détruite) sous forme de matrice board_size * board_size * 4
        removed_elements (np.ndarray) : les tuiles à replacer pour reconstruire la solution
        return_pos (bool) : si vrai, on retourne en plus une liste indiquant où à été replacé chaque pièce (utilisé dans reconstruct_local_search)

    Returns:
        reconstructed_sol (np.ndarray) : la solution reconstruite
        pos (List[np.ndarray, np.ndarray]) : la liste indiquant les coordonnées (deuxième élément) où chaque pièce (premier élément) a été placée
    """
    board_size = solution.shape[0]
    reconstructed_sol = solution.copy()
    to_reassign = np.where(reconstructed_sol[:, :, 0] == -1)
    nonzero_per_removed_tile = np.count_nonzero(removed_elements, axis=1)

    if return_pos:
        Pos = []

    for e in np.transpose(to_reassign):
        if (np.allclose(e, np.array([0, 0])) or
            np.allclose(e, np.array([0, board_size-1])) or
            np.allclose(e, np.array([board_size-1, 0])) or
            np.allclose(e, np.array([board_size-1, board_size-1]))):
            # Dans un coin
            i = np.where(nonzero_per_removed_tile == 2)[0][0]

        elif (e[0] == 0 or 
              e[0] == board_size-1 or
              e[1] == 0 or
              e[1] == board_size-1):
            # Sur un bord
            i = np.where(nonzero_per_removed_tile == 3)[0][0]

        else:
            # milieu
            i = np.where(nonzero_per_removed_tile == 4)[0][0]
        
        reconstructed_sol[e[0], e[1]] = greedy_replace(reconstructed_sol, removed_elements[i], e)

        if return_pos:
            Pos.append([removed_elements[i], e])

        nonzero_per_removed_tile = np.delete(nonzero_per_removed_tile, i, 0)
        removed_elements = np.delete(removed_elements, i, 0)

    if return_pos:
        return reconstructed_sol, Pos
    
    return reconstructed_sol



    
def reconstruct_local_search(solution, removed_elements, puzzle):
    """
    """
    # On reconstruit de façon gloutonne en mettant dans le bon type de case (bords dans bords etc.)
    current_sol = solution.copy()
    current_sol, Pos = reconstruct_deterministic_greedy(current_sol, removed_elements, puzzle,return_pos=True)
    current_n_conflict = evaluate(puzzle, current_sol)
    reconstructed_sol, best_n_conflict = current_sol, current_n_conflict

    n_removed_elements = removed_elements.shape[0]
    board_size = solution.shape[0]

    for _ in range(10):
        # Restarts
        current_sol, Pos = reconstruct_deterministic_greedy(solution, removed_elements, puzzle,return_pos=True)
        current_n_conflict = evaluate(puzzle, current_sol)

        best_solution_restart, best_n_conflict_restart = current_sol, current_n_conflict

        for iter in range(400):
            # Sélection aléatoire d'un voisin selon solver_local_search.deux_swap_un_seul_valide
            # Code sensiblement pareil jusqu'à 380.
            neigh = current_sol.copy()

            # On choisit 2 pièces à échanger dont une parmi celles qu'on a replacé (removed_elements)
            valid = False
            corner_or_edge = False
            while not valid:
                # On tire une pièce qu'on a replacé
                i = np.random.randint(n_removed_elements)
                tile1, pos1 = Pos[i]

                # On tire une autre pièce
                pos2 = pos1
                while np.allclose(pos1, pos2):
                    pos2 = np.random.randint(0, board_size, 2)
                tile2 = neigh[pos2[0], pos2[1]]

                # On garde cette sélection si les deux pièces sont de la même géométrie (bords, coins)
                if np.count_nonzero(tile1) == np.count_nonzero(tile2):
                    valid = True
                    if np.count_nonzero(tile1) != 4:
                        corner_or_edge = True
            
            if corner_or_edge:
                # On échange les deux bords/coins en échangeant leurs rotations
                corn_edge_tile_1 = np.where(np.array(neigh[pos1[0], pos1[1]]) == 0)[0]
                corn_edge_tile_2 = np.where(np.array(neigh[pos2[0], pos2[1]]) == 0)[0]

                if np.allclose(corn_edge_tile_1, corn_edge_tile_2):
                    # Même orientation (bords du même côté du plateau)
                    neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = current_sol[pos2[0], pos2[1]], current_sol[pos1[0], pos1[1]]

                else:
                    if len(corn_edge_tile_1) == 1:
                        # Les deux pièces sont des bords
                        if corn_edge_tile_1[0] + corn_edge_tile_2[0] == 1 or corn_edge_tile_1[0] + corn_edge_tile_2[0] == 5:
                            # Les indices sont 0 et 1, ou 2 et 3 (bords opposés)
                            neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[2], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[2]

                        elif corn_edge_tile_1[0] + corn_edge_tile_2[0] == 2:
                            # Bords haut et gauche
                            if corn_edge_tile_1[0] == 0:
                                # Pièce 1 est un bord haut : on doit la tourner de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]
                            else:
                                # Pièce 1 est un bord gauche : on doit la tourner de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]

                        elif corn_edge_tile_1[0] + corn_edge_tile_2[0] == 4:
                            # Bords bas et droite
                            if corn_edge_tile_1[0] == 3:
                                # Pièce 1 est un bord droit : on doit la tourner de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]
                            else:
                                # Pièce 1 est un bord bas : on doit la tourner de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]
                        
                        elif abs(corn_edge_tile_1[0] + corn_edge_tile_2[0] == 4) == 3:
                            # Bords haut et droite
                            if corn_edge_tile_1[0] == 0:
                                # Pièce 1 est un bord haut : on doit la tourner de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]
                            else:
                                # Pièce 1 est un bord droit : on doit la tourner de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]
                        
                        else:
                            # Bords bas et gauche
                            if corn_edge_tile_1[0] == 1:
                                # Pièce 1 est un bord bas : on doit la touner de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]
                            else:
                                # Pièce 1 est un bord droit : on doit la tourner de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]
                    
                    else:
                        # Les deux pièces sont des coins
                        if np.sum(corn_edge_tile_1 == corn_edge_tile_2) == 0:
                            # Les deux pièces sont dans les coins opposés ; rotation de 180°
                            neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[2], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[2]

                        elif corn_edge_tile_1[0] == corn_edge_tile_2[0]:
                            # np.where est séquentiel : les indices sont triés. 
                            # Premiers éléments égaux <=> les premiers éléments sont 0 (coins haut-G et haut-D) ou 1 (coins bas-G et bas-D)
                            if np.sum(corn_edge_tile_1) == 3:
                                # On doit tourner la pièce 1 de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]
                            else:
                                # On doit tourner la pièce 2 de 270°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]
                        
                        else:
                            # Premiers éléments diff <=> les deuxièmes éléments sont 2 (coins haut-G et bas-G) ou 3 (coins haut-D et bas-D)
                            if np.sum(corn_edge_tile_1) == 3:
                                # On doit tourner la pièce 1 de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[3], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[1]
                            else:
                                # On doit tourner la pièce 2 de 90°
                                neigh[pos1[0], pos1[1]], neigh[pos2[0], pos2[1]] = puzzle.generate_rotation(current_sol[pos2[0], pos2[1]])[1], puzzle.generate_rotation(current_sol[pos1[0], pos1[1]])[3]

            else:
                neigh[pos1[0], pos1[1]] = greedy_replace(current_sol, current_sol[pos2[0], pos2[1]], [pos1[0], pos1[1]])
                neigh[pos2[0], pos2[1]] = greedy_replace(current_sol, current_sol[pos1[0], pos1[1]], [pos2[0], pos2[1]])
            
            # On remet à jour où se trouve désormais la tuile (de removed_elements)
            Pos[i][1] = pos2

            # Recuit simulé
            neigh_conflicts = evaluate(puzzle, neigh)
            delta = neigh_conflicts - current_n_conflict

            if delta <= 0:# or np.random.rand() < np.exp(-delta/temp):
                current_sol, current_n_conflict = neigh, neigh_conflicts

            if current_n_conflict < best_n_conflict_restart:
                best_solution_restart, best_n_conflict_restart = current_sol, current_n_conflict

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            reconstructed_sol = best_solution_restart


    return reconstructed_sol


def evaluate(puzzle, solution):
    """
    Évalue la solution encodée sous forme de matrice, en la remettant en forme souhaitée par le vérificateur de solutions.

    Args:
        puzzle (EternityPuzzle) : l'instance du puzzle en cours de résolution
        solution (np.ndarray) : l'encodage de la solution sous forme de matrice board_size * board_size * 4
    
    Returns:
        _ (int) : le nombre de conflits dans la solution
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

    time_credit = 1800

    #Initialisation des fonctions de destruction
    destruction_functions = [destroy, destroy_worst_tiles]

    #Initialisation des fonctions de reconstruction
    reconstruction_functions = [reconstruct_randomized_greedy, reconstruct_deterministic_greedy, reconstruct_local_search]
    
    #Initalisation des poids de mise à jour
    psi = [1,0.8,0.6,0.1]
    lambda_w = 0.97

    # Recherche
    while ((time()-t0) + iteration_duration) < time_credit - 5:
        #Initialisaiton de poids pour l'ALNS
        rho_dest = np.array([3.,1.])
        rho_rec = np.array([1. for truc in reconstruction_functions])

        # Temps de début du restart
        t1 = time()

        puzzle = copy.deepcopy(eternity_puzzle)
        # Restart aléatoire depuis la solution initiale
        random.shuffle(puzzle.piece_list)

        current_solution, current_n_conflict = solve_heuristic(puzzle)
        current_solution = np.flip(np.array(current_solution).reshape((n, n, 4)), 0)
        best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict

        temp = 10
        d = 0.1
     
        for _ in range(1000):
            # Choix fonction de destruction
            total_weight_destruction = np.sum(rho_dest)
            
            destruction_index = random.choices(range(len(rho_dest)), weights=rho_dest / total_weight_destruction, k=1)[0]
            
            destruction_function = destruction_functions[destruction_index]

            # Choix fonction de reconstruction
            total_weight_reconstruction = np.sum(rho_rec)
            reconstruction_index = random.choices(range(len(rho_rec)), weights=rho_rec / total_weight_reconstruction, k=1)[0]
            reconstruction_function = reconstruction_functions[reconstruction_index]

            solution = reconstruction_function(*destruction_function(current_solution, d), puzzle)

            # Acceptation éventuelle de la solution reconstruite
            n_conflict = evaluate(puzzle, solution)
            delta = n_conflict - current_n_conflict

            if delta <= 0:# or np.random.rand() < np.exp(-delta/temp):
                index_maj = 1

                current_solution, current_n_conflict = solution, n_conflict

            else:
                index_maj = 3

            if current_n_conflict < best_n_conflict_restart:
                index_maj = 0

                best_solution_restart, best_n_conflict_restart = current_solution, current_n_conflict

            # MÀJ des poids de selection de fonctions

            rho_dest[destruction_index] = lambda_w*rho_dest[destruction_index] + (1-lambda_w)*psi[index_maj]

  
            rho_rec[reconstruction_index] = lambda_w*rho_rec[reconstruction_index] + (1-lambda_w)*psi[index_maj]

            
            temp *= 0.99

        if best_n_conflict_restart < best_n_conflict:
            best_n_conflict  = best_n_conflict_restart
            best_solution = best_solution_restart

            print(best_n_conflict)
        
        if best_n_conflict == 0: break
        
        iteration_duration = time() - t1

    log_execution(dict_board_size_instance[eternity_puzzle.board_size], best_n_conflict, time()-t0)

    # Mise en forme pour la visualisation
    best_solution = [
        tuple(e) for e in np.reshape(np.flip(best_solution, axis=0), newshape=(-1, 4))
    ]

    return  best_solution, best_n_conflict