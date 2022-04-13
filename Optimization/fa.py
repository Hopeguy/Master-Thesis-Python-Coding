from math import exp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import intelligence


class fa(intelligence.sw):
    """
    Firefly Algorithm
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, csi=1, psi=1,
                 alpha0=1, alpha1=0.1, norm0=0, norm1=0.1):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param csi: mutual attraction (default value is 1)
        :param psi: light absorption coefficient of the medium
        (default value is 1)
        :param alpha0: initial value of the free randomization parameter alpha
        (default value is 1)
        :param alpha1: final value of the free randomization parameter alpha
        (default value is 0.1)
        :param norm0: first parameter for a normal (Gaussian) distribution
        (default value is 0)
        :param norm1: second parameter for a normal (Gaussian) distribution
        (default value is 0.1)
        """

        super(fa, self).__init__()

        self.__agents = np.random.randint(lb, ub, (n, dimension)) #Initilaze the starting population, random ints between lb and ub, agents times 2
        
        self._points(self.__agents)
        
        Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()] #Returns the solution with the lowest fittness score
        Gbest = copy.deepcopy(Pbest) # takes the best (lowest) fitness value and assigne it to Gbest variable
    
        for t in range(iteration):      #Loop that goes equalt to number of iterations
            
            alpha = alpha1 + (alpha0 - alpha1) * exp(-t)    #Alpha is randomization parameter, that becomes smaller the more iterations
            #print("alpha value: ",alpha, " iteration: ", t)   #Too see how alpha changes each iteration
            for i in range(n): # For each agent we go into the for loop
                fitness = [function(x) for x in self.__agents] #Updates the fitness for each agent after after agent "i" have been moved by all other agents
                #This could probably be better written as it needs to redo all agents when only one have been edited [optimize so it only edits in fitness list the one that have changed pos]
                for j in range(n): #each agent are compared to each other agent
                    if fitness[i] > fitness[j]:
                        
                        self.__move(i, j, t, csi, psi, alpha, dimension, ub, lb)
                        
                                    
                    else:
                        
                        rand_value = (alpha * abs(ub-lb) * (np.random.rand(1,dimension) - 0.5))
                        rand_fill = []
                        for v in rand_value:
                            for w in v:
                                rand_fill.append(w)
                        
                        self.__agents[i] = list(map(lambda a,b: a+b, list(map(int, rand_fill)), self.__agents[i])) 
                        
            
            self.__agents = np.clip(self.__agents, lb, ub)  #Checks that all agents are in the required bound, and change them to the highest or lowest if not (lb, or ub)
            
            self._points(self.__agents)
            
            Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
            #print("Pbest before we compare with Gbest", Pbest)
            if function(Pbest) < function(Gbest):
                
                Gbest = copy.deepcopy(Pbest)

            print(Pbest, "best solution for iteration: ", t, "With fitness value: ", function(Pbest))    
        self._set_Gbest(Gbest)

    def __move(self, i, j, t, csi, psi, alpha, dimension, ub, lb):

        r = np.linalg.norm(self.__agents[i] - self.__agents[j]) #Gives the Forbenius norm, "same as cartesian distance"
        beta = csi / (1 + psi * (r ** 2)) #Approximative method to Changes the attractivness between the two agents depending on their distance r and mutual attraction cis
       
        rand_value_move = alpha * abs(ub-lb) * (np.random.rand(1,dimension) - 0.5)
        rand_fill_move = []
        for v in rand_value_move:
            for w in v:
                rand_fill_move.append(w)
        attract_list = list(map(lambda a,b: a+b, list(map(int, rand_fill_move)), self.__agents[i]))
        beta_walk = beta * (self.__agents[i] - self.__agents[j])
        self.__agents[i] = list(map(lambda a,b: a+b, attract_list, beta_walk))

            