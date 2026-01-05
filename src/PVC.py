from enum import Enum
from heapq import heappush, heappop
from math import inf
from bigtree import Node
import numpy as np
from src.heap import Heap

class PVC:
    """
        Probleme de Voyageur de Commerce,

        Soit S un ensemble fini (dans cette implementation on
        considere S sous-ensemble de N)

        Soit w une function, i.e
        w : S x S -> R, i.e : R l'ensemble des réels  
            (u, v) |-> w(u, v)

        Soit A un ensemble d'uplets, i.e
        A = {(u, v, w(u, v))/ u $\\in$ S et v $\\in$ S)}
    
        et soit
        G = (S, A, w) un graphe non orienté et valué.
        S, A et w sont resp. l'ensemble des sommets, d'arretes 
        et la fonction du poids de G.     

        Attributs:
            sommets (set) : l'ensemble des sommets de G
            arretes (list) : l'ensemble des arretes de G
            taille (int) : le nombre de sommets de G
            M (ndarray(n, n)) : La matrice de distances de G    
            adj_dict (dict) : liste d'adjacence de G
                {sommet : [adjacents_du_sommet]}    
    """

    # Constructor
    def __init__(self):
        # inner structure
        self.sommets_ = set() 
        self.arretes_ = []
        self.taille_ = 0

        # representations
        self.M = None
        self.adj_dict_ = None

    @staticmethod
    def de_matrice_a_liste(M):
        """
            Transforme la matrice des distances en
            liste d'adjacence

            Args:
                M (ndarray(n, n)) : La matrice des distances

            Returns:
                dictM (dict) : Liste d'adjacence
        """
        n = M.shape[0]
        # Initialisation of the dict
        dictM = {i : [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] < np.inf:
                    dictM[i].append(j)
                    dictM[j].append(i)
        return dictM

    def charger_de_matrice(self, M):
        assert M.shape[0] == M.shape[1], "M must be a square matrix" 
        assert np.allclose(M, M.T), "La matrice des distances doit être symétrique"

        self.taille_ = M.shape[0]
        self.sommets_ = set(range(self.taille_))
        self.arretes_ = [(i, j, M[i, j]) for i in range(self.taille_) \
                    for j in range(i + 1, self.taille_) if M[i, j] < np.inf]
            
        M_copy = M.copy()
        self.M = M_copy

        self.adj_dict_ = self.de_matrice_a_liste(self.M)

    @staticmethod
    def draw_forest(pi):
        def build_tree(root, tree):
            parent = Node(str(root + 1))
            if tree.get(root) is None:
                return parent
            for c in tree[root]:
                child = build_tree(c, tree)
                child.parent = parent  
            return parent
        
        children = dict()
        forest = []
        for i, v in enumerate(pi):
            if v < 0:
                forest.append(i)
                children.setdefault(i, [])
            else:   
                children.setdefault(v, []).append(i)

        for root in forest:
            build_tree(root, children).vshow() 
            print("__________")

    def prim(self, source=0, verbose=False):
        """
            Une implementation efficace de l'algorithme de Prim
            pour trouver un arbre de poids minimum d'un graphe G
            en utilisant un tas de type min, où la priorité dans
            ce tas est défini par le poids d'arrete qui relie le
            sommet en question avec l'ACM, np.inf sinon

            Args:
                source (int, optionnel) : The node of the source of the shortest path.
                Default 0
                verbose (boolean, optionnel) : Return the intermediate 
                views of the heap
                Default False

            Raises:
                ValueError if the source node source doesn't belong to the nodes set
                AssertionError if the Distance matrix contains negative values

            Returns:
                pi (ndarray(n)) : The array of predecessors of the ACM 

            Abreviation:
                ACM : Arbre Couvrant de Poids Minimum   
        """ 
        assert self.taille_ > source >= 0 , "source out of range" 

        # la fonction de poids
        w_ = {}
        for u, v, w in self.arretes_:
            w_[(u, v)] = w
            w_[(v, u)] = w

        H = Heap(type='min')
        # Initialise the distances and predecessors arrays
        d = np.ones(self.taille_) * np.inf
        d[source] = 0
        pi = -np.ones(self.taille_, dtype=int)

        # Initialise the 'min' heap
        tas = zip(set(range(self.taille_)), d)
        H.init_heap(tas)
        while not H.is_empty():
            # get the nearest vertex to the MST
            curr_vertex, curr_priority = H.dequeue()

            # the successors of curr_vertex
            succ_curr_vertex = self.adj_dict_[curr_vertex]
            if verbose:
                print(f"current vertex :\n{curr_vertex} -> {succ_curr_vertex}")

            for vertex in succ_curr_vertex:
                # Check if vertex is in the heap
                # and if we can improve its estimation  
                if vertex in H.positions_ and d[vertex] > w_[curr_vertex, vertex]:
                    # Update the estimation
                    d[vertex] = w_[curr_vertex, vertex]
                    pi[vertex] = curr_vertex

                    # Update the heap
                    H.update_priority(vertex, d[vertex])
                    if verbose:
                        H.show() 
        return pi, d

    def backtrack(self):
        i = [0]
        colors = []
        pi = []
        P = []
        S = []
        P_ = []
        S_ = []

        class Etat(Enum):
            """États de découverte des sommets pour le parcours en profondeur"""
            WHITE = 1 # pas encore découvert
            GRAY = 2 # au cours d'exploration
            BLACK = 3 # exploré

        def visiter_en_profondeur(source, colors, P, S, P_, S_, pi, i):
            colors[source] = Etat.GRAY
            i[0] += 1
            # Début de l'exploration du sommet source
            P[source] = i[0]
            P_.append(source) 

            # parcourir les successeurs de source
            for u in self.adj_dict_.get(source):
                # si pas encore découvert
                if colors[u] == Etat.WHITE:
                    # sourcee est le predecesseur de u dans l'arborescence d'exploration
                    pi[u] = source 
                    # explorer u
                    visiter_en_profondeur(u, colors, P, S, P_, S_, pi, i)

            # Fin de l'exploration du sommet source 
            colors[source] = Etat.BLACK
            i[0] += 1
            S[source] = i[0] 
            S_.append(source) 
            return pi, P, S
        
        for _ in range(len(self.sommets_)):
            colors.append(Etat.WHITE)
            pi.append(-1)
            P.append(0)
            S.append(0)

        for u in self.sommets_:
            if colors[u] == Etat.WHITE:
                visiter_en_profondeur(u, colors, P, S, P_, S_, pi, i)    
        return pi, P_, S_, P, S

    def longueur(self, c):
        l = 0
        for i in range(len(c) - 1):
            l += self.M[c[i], c[i + 1]]

        return l

    def plus_proche_distance(self, c, visited):
        min_dist = inf
        u = -1
        v = -1

        for i in self.sommets_:
            if not visited[i]:
                for j, q in enumerate(c):
                    d = self.M[i, q]
                    if d < min_dist:
                        # u est le point le plus proche dans S \ c
                        # q est le point le plus proche dans le cycle c 
                        # il se trouve a l'indice v
                        min_dist = d
                        u, v = i, j
 
        visited[u] = True                 
        return u, v

    def PPP(self, source=0):
        c = [source]
        visited = [False] * self.taille_
        visited[source] = True
        while len(c) != self.taille_:
            u, v = self.plus_proche_distance(c, visited)
            c.insert(v + 1, u)
        c.append(c[0])    

        return np.array(c) 

    def OptPPP(self, c):
        # c un cycle obtenu par PPP
        c = list(c) 
        c.pop()
        n = len(c)
        amelioration = True

        while amelioration:
            amelioration = False
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b = c[i], c[i + 1]
                    d, e = c[j], c[j + 1]
                    cost_before = self.M[a, b] + self.M[d, e]
                    cost_after = self.M[a, d] + self.M[b, e]

                    if cost_after < cost_before:
                        # Décroisement : inversion du segment
                        c[i + 1: j + 1] = list(reversed(c[i + 1: j + 1]))
                        amelioration = True          
        c.append(c[0])
        return np.array(c)

    def OptPrim(self):
        source = np.random.randint(0, self.taille_)
        pi, d = self.prim(source)
        #self.draw_forest(pi)
        # construire l'arbre
        M = np.ones(shape=(self.taille_, self.taille_), dtype=float) * np.inf
        for i, v in enumerate(pi):
            if v == -1:
                continue
            M[v, i] = d[i]
            M[i, v] = d[i]

        ACM = PVC()
        ACM.charger_de_matrice(M)
        # trouver l'ordre préfixe
        _, P_, _, _, _ = ACM.backtrack()
        c = P_
        c.append(P_[0]) 

        return np.array(c)
    
    def heuristique_demi_somme(self, cycle):
        """
        Calcule la borne de l'heuristique de la demi-somme.
        Pour chaque sommet, on prend la demi-somme des deux plus petites arêtes incidentes.
        """
        n = len(cycle)
        bound = 0  
        # les sommets ne faisant pas partie du cycle
        rest = [r for r in range(self.taille_) if r not in cycle or n == 1]
        for i in rest:
            distances = sorted(d for d in self.M[i] if d < np.inf)
            bound += distances[1] + distances[2]

        # le cout fixé
        for i in range(1, n - 1):
            bound += self.M[cycle[i - 1]][cycle[i]] + self.M[cycle[i]][cycle[i + 1]]

        # le premier et dernier element 
        # partiellement contraint
        if n > 1:
            for v, i in zip([self.M[cycle[0]][cycle[1]], self.M[cycle[-2]][cycle[-1]]], [0, -1]):
                bound += v
                distances = sorted(d for d in self.M[cycle[i]] if d < np.inf)
                if distances[1] == v:
                    bound += distances[2]
                else:    
                    bound += distances[1]
        return bound / 2 

    def HDS(self):
        """
        Algorithme de recherche par heuristique de la demi-somme.
        Prend en entrée un graphe complet valué (matrice d'adjacence).
        Retourne une tournée hamiltonienne de coût minimal.
        """
        best_cost = inf
        best_solution = None
        
        # File de priorité pour explorer les nœuds avec la meilleure borne
        # Chaque élément est (borne, coût_actuel, chemin)
        heap = []
        
        # Commencer du sommet 0   
        # Calcul de la borne initiale
        initial_bound = self.heuristique_demi_somme([0])
        path = [0]
        heappush(heap, (initial_bound, 0, path))
        
        while heap:
            bound, current_cost, path = heappop(heap)        
            # Élagage : si la borne est déjà pire que la meilleure solution
            if bound >= best_cost:
                continue
                
            # Si on a une solution complète
            if len(path) == self.taille_:
                # Ajouter le retour au point de départ
                complete_cost = current_cost + self.M[path[-1], path[0]]
                if complete_cost < best_cost:
                    best_cost = complete_cost
                    best_solution = path + [path[0]]
                continue

            for next_node in range(self.taille_):
                if not next_node in path:
                    # Calculer le nouveau coût
                    new_cost = current_cost + self.M[path[-1], next_node]               
                    new_path = path + [next_node]
                    # Calculer la nouvelle borne
                    new_bound = self.heuristique_demi_somme(new_path)
                    
                    # Ajouter à la file de priorité si prometteur
                    if new_bound < best_cost:
                        heappush(heap, (new_bound, new_cost, new_path))
        
        return best_solution
