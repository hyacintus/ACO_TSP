import random
import numpy as np

def RouletteWheelSelection(P):

    r = random.uniform(0, 1)
    C = np.cumsum(P)

    # get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    # Numeri = get_indexes(r, C)
    Indici_Minori = np.where(C <= r)
    j = max(Indici_Minori[1]) + 1
    # Quando ci si muove con le matrici si devono indicare tutti e 2 gli indici, se no ti da errore
    # j = C[0, Massimo+1]
    return j


