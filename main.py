import pandas as pd  # Libreria per aprire .csv e .xlxs
import numpy as np
import seaborn as sns # lavora come wrapping di matplot
import matplotlib.pyplot as plt
from RouletteWheelSelection import RouletteWheelSelection

# Commento aggiunto da DAAIL

# ndarray = np.array([[5, np.inf, 15, 45], [9, np.inf, 11, 60],
#                     [16, 10, 19, 70],
#                     [18, 26, 20, np.inf],
#                     [20, 7, 21, np.inf]])
#
# print('Infinite value in array:', ndarray[np.isinf(ndarray)])
#
# ndarray[np.isinf(ndarray)] = 0
#
# print(ndarray)

# Variabile D = DataFrame
D = pd.read_excel('Archi.xlsx', header=None, skiprows=1, nrows=11, usecols='B:L')
# -header=None serve a dire: Non utilizzare i nomi della colonna come indici
# -skiprows=1 --> salta una righa
# -nrows=11 --> Prendi 11 colonne
# -usecols='B,L' --> prendi dalla colonna B alla L
print(D.head())  # Stampo la testa del mio dataset, di solito sono 5 righe

nVar = len(D)  # Numero di Nodi
# print(nVar)
Nodo_Partenza = 0

# PARAMETRI ACO
MaxIt = 10                              # Massimo numero di iterazioni
nAnt = 10                               # Numero di formiche (Dimensione della popolazione)
Q = 1
# D = np.matrix(D)
# media = D.mean()
# media = np.matrix(D).mean()
tau0 = 10*Q/(nVar*np.matrix(D).mean())     # Feromone iniziale: 10*1/(20*media_di_tutte_le_distanze) = INTENDSITA' DELLA PISTA
alpha = 1                               # Peso esponenziale del feromone (Phromone Exponential Weight)
beta = 1                                # Peso esponenziale euristico (Heuristic Exponential Weight)
rho = 0.05                              # Tasso di evaporazione (Evaporation Rate)

# INIZIALIZZAZIONE
D = np.matrix(D)
eta = np.matrix(1/D)                               # Matrice di informazioni euristiche (Heuristic Information Matrix) = VISIBILITA' = INVERSO DELLA DISTANZA
eta[np.isinf(eta)] = 0
tau = np.matrix(tau0*np.ones((nVar,nVar)))
BestCost = np.zeros((MaxIt,1))              # Matrice che contiene i migliori valori di costo


# Struttura globale Ant
class Ant:
    def __init__(self, Tour, Cost):
        self.Tour = Tour
        self.Cost = Cost


# Formiche di ogni iterazione
Winner_Ant = Ant([], [])
BestSol = Ant([], float('inf'))
#ant = [Ant([], 0) for i in range(nAnt)]  # Conta automaticamente il -1
# Sarebbe come scrivere:
# ant = []
# for i in range(0, 100):
#     ant.append(Ant([], 0))

# CICLO PRINCIPALE ACO
for it in range(MaxIt):
    ant = [Ant([], 0) for i in range(nAnt)]
    for k in range(nAnt):
        ant[k].Tour.append(Nodo_Partenza)
        for l in range(1,nVar):
            i = ant[k].Tour[-1]
            P = np.multiply(np.power(tau[i, :], alpha), np.power(eta[i, :], beta)) # Calcolo probabilità di andare in un certo nodo
            P[0, ant[k].Tour] = 0
            Somma_P = P.sum()
            P = np.divide(P, Somma_P)
            j = RouletteWheelSelection(P)    # Nodo di arrivo
            ant[k].Tour.append(j)            # Aggiorno Tour
            ant[k].Cost = ant[k].Cost + D[ant[k].Tour[-2], ant[k].Tour[-1]]       # Aggiorno Costo

        # faccio tornare indietro la formica
        ant[k].Tour.append(ant[k].Tour[0])
        ant[k].Cost = ant[k].Cost + D[ant[k].Tour[-2], ant[k].Tour[-1]]

        # Aggiornamento della soluzione migliore
        if ant[k].Cost < BestSol.Cost:
            BestSol = ant[k]

    # Formica Vincitrice
    if it == 0:
        Winner_Ant = BestSol
    if it != 0 and BestSol.Cost < Winner_Ant.Cost:  # Salva in BestSol solo la formica con il 'costo' minore
        Winner_Ant = BestSol

    # Aggiornamento del Feromone
    for k in range(nAnt):
        tour_single_ant = np.matrix(ant[k].Tour)
        # tour_single_ant = np.concatenate((np.matrix(ant[k].Tour), np.matrix(ant[k].Tour[0])), axis=1)
        # tour_2 = np.concatenate((tour, np.matrix(ant[k].Tour[0])), axis=1)
        for n in range(nVar):
            i = tour_single_ant[0,n]
            j = tour_single_ant[0,n+1]
            tau[i,j] = tau[i, j] + Q/ant[k].Cost

    # Evaporazione Feromone
    tau = (1 - rho) * tau

    # Conservo il costo migliore dell'iterazione
    BestCost[it] = BestSol.Cost

print("Formica vincitrice:")
print("Tour:" + str(Winner_Ant.Tour))
print("Cost:" + str(Winner_Ant.Cost))

BestCost = np.matrix(BestCost).transpose()
Iterazioni = np.matrix(range(1, it+2))
df = np.concatenate((Iterazioni,BestCost), axis=0).transpose()

df = pd.DataFrame(df, columns=['Iterations', 'BestCost'])
g = sns.lineplot(data=df, x='Iterations', y='BestCost')
plt.show()  # Per far vedere il grafico a schermo




















