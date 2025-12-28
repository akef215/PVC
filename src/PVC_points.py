import numpy as np
import matplotlib.pyplot as plt
from src.PVC import PVC

class PVC_points(PVC):
    def __init__(self):
        super().__init__()
        self.points_ = None
        self.idx_map_ = None
        self.inv_ = None         

    def charger_de_liste(self, points):
        self.__init__()
        self.points_ = points

        M = np.zeros(shape=(len(points), len(points)), dtype=float)
        idx_map, inv = dict(), dict()
        for i, p in enumerate(points):
            idx_map[p] = i
            inv[i] = p
            for j, q in enumerate(points):     
                M[i, j] = self.euclidean_distance(p, q)

        self.charger_de_matrice(M)
        self.idx_map_ = idx_map
        self.inv_ = inv

    def charger_de_fichier(self, fichier):
        with open(fichier, "r") as file:
            points = []
            for line in file:
                line = line.strip()
                if not line:
                    continue
                line = line.lstrip("(").rstrip(")")
                x, y = map(float, line.split(","))
                points.append((x, y))

            self.charger_de_liste(points)        

    @staticmethod
    def euclidean_distance(P, Q):
        P = np.array(P)
        Q = np.array(Q)
        return np.sqrt(np.sum((P - Q)**2))     

    @staticmethod
    def plot_cycle(cycle, titre="Stratégie du point le plus proche"):
        plt.title(titre)
        plt.plot(np.array(cycle).T[0], np.array(cycle).T[1], 'rx--')
    
    def _map_cycle(self, cycle):
        """Retourne la liste des points à partir des indices"""
        return [self.inv_[i] for i in cycle]
    
    def longueur(self, c):
        assert all(p in self.idx_map_ for p in c), "Tous les points doivent exister dans idx_map_"
        cycle = [self.idx_map_[i] for i in c]
        return super().longueur(cycle)

    def PPP(self, source):
        assert self.taille_ > 2, "le nombre de points doit etre superieur a 3" 
        assert source in self.points_, "Il faut choisir un point existant"
        s = self.idx_map_[source]
        cycle = super().PPP(s)
        return self._map_cycle(cycle)

    def OptPPP(self, source):
        assert self.taille_ > 2, "le nombre de points doit etre superieur a 3" 
        assert source in self.points_, "Il faut choisir un point existant"
        s = self.idx_map_[source]
        cycleB = super().PPP(s)
        cycle = super().OptPPP(cycleB)
        return self._map_cycle(cycle)

    def OptPrim(self):
        assert self.taille_ > 2, "le nombre de points doit etre superieur a 3" 
        cycle = super().OptPrim()
        return self._map_cycle(cycle)
    
    def HDS(self):
        assert self.taille_ > 2, "le nombre de points doit etre superieur a 3" 
        cycle = super().HDS()
        return self._map_cycle(cycle)
