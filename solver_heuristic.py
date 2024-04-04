import copy
import numpy as np
from eternity_puzzle import EternityPuzzle


GRIS = 0
NOIR = 23
ROUGE = 24
BLANC = 25

NORD = 0
SUD = 1
OUEST = 2
EST = 3




def solve_heuristic(eternity_puzzle):
    """
    Heuristic solution of the problem
    :param eternity_puzzle: object describing the input
    :return: a tuple (solution, cost) where solution is a list of the pieces (rotations applied) and
        cost is the cost of the solution
    """
    # TODO implement here your solution
    solution = [(23,23,23,23)] * eternity_puzzle.board_size**2
    remaining_piece = copy.deepcopy(eternity_puzzle.piece_list)

    

    # shuffle the pieces
    # take the orientation that works the best
    for i in range(eternity_puzzle.n_piece):

        piece = choix_piece(i//eternity_puzzle.board_size,i%eternity_puzzle.board_size,solution,remaining_piece,eternity_puzzle)
        
        best_n_conflicts = 2**32
        new_piece = None

        for k in range(0,4):

            piece_permuted = eternity_puzzle.generate_rotation(piece)[k]
            temp = solution.copy()
            temp[i] = piece_permuted
            cost = eternity_puzzle.get_total_n_conflict(temp)
            if  cost < best_n_conflicts:
                new_piece = piece_permuted
                best_n_conflicts = cost


        solution[i] = new_piece

        remaining_piece.remove(piece)

    return solution, eternity_puzzle.get_total_n_conflict(solution)



def choix_piece(i,j,solution,pieces,puzzle:EternityPuzzle):

    """
    Args:
        i (int) : ligne de la tuile
        j (int) : colonne de la tuile
        solution (List[pieces]) : solution actuelle
        pieces (List[pieces]): pieces disponibles
        puzzle (EternityPuzzle) : instance du problème

    Return :
        piece (piece) : une piece du bord ou du centre en fonction des coordonnées données
    
    """

    board_size = puzzle.board_size


    if len(pieces)==1:
        return pieces[0]
    
    k = board_size * i + j
    k_est = board_size * i + (j + 1)
    k_ouest = board_size * i + (j - 1)
    k_sud = board_size * (i - 1) + j
    k_nord = board_size * (i + 1) + j

    # On verifie si on doit renvoyer une piece du bord
    if i == 0 or j==0 or i == board_size-1 or j == board_size -1:
        pieces_possibles = [piece for piece in pieces if GRIS in piece]

    else :
        pieces_possibles = [piece for piece in pieces if not GRIS in piece]

    scores = []

    if i==0:
        c_sud = GRIS

    else : 
        c_sud = solution[k_sud][NORD]
    
    if j==0:
        c_ouest = GRIS

    else:
        c_ouest = solution[k_ouest][EST]

    if i == board_size -1:
        c_nord = GRIS
    else :
        c_nord = solution[k_nord][SUD]

    if j == board_size -1:
        c_est = GRIS
    else :
        c_est = solution[k_est][OUEST]


    couleurs_voisins = (c_nord,c_sud,c_ouest,c_est)

    #print(i,j,couleurs_voisins,pieces_possibles,pieces)
    for piece in pieces_possibles:

        
        best_score_piece = 0

        for k in range(0,4):
            
            piece_permuted = puzzle.generate_rotation(piece)[k]
            score = points_communs(piece_permuted,couleurs_voisins)
            if score > best_score_piece:
                best_score_piece = score

        scores.append(best_score_piece)
    
    
    best = np.argmax(scores)
    
    if isinstance(best,np.ndarray):
        best_piece = pieces_possibles[best[0]]
    else:
        best_piece = pieces_possibles[best]

 
    return best_piece



def points_communs(tuple1,tuple2):


    score = 0

    for i in range(len(tuple1)):
        if tuple1[i] == tuple2[i]==0:
            score += 2

        elif tuple1[i] == tuple2[i]:
            score +=1
        elif tuple1[i] == 0 :
            score -= 1


    return score

    


