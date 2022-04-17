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
        fitness = [function(x) for x in self.__agents]


        for t in range(iteration):      #Loop that goes equalt to number of iterations
            
            alpha = alpha1 + (alpha0 - alpha1) * (exp(-t/4))    #Alpha is randomization parameter, that becomes smaller the more iterations
            #print("alpha value: ",alpha, " iteration: ", t)   #Too see how alpha changes each iteration
            for i in range(n): # For each agent we go into the for loop
                
                #print(fitness, "fitness list")
                #print(self.__agents, "Agents")
                #print(self.__agents[i], "Ith Agent")
                for j in range(n): #each agent are compared to each other agent
                    if fitness[i] > fitness[j]: #if agent i are higher (we want to minimize) we move agent i towards agent j with move function 
                        
                        self.__move(i, j, t, csi, psi, alpha, dimension, ub, lb) #Calls move function an changes the agent i position depending on agent j position
                        
                                    
                    else: #If fitness of I is higher J then it moves randomly as we are always only looking at each pair of fireflies
                        
                        rand_value = (alpha * abs(ub-lb) * (np.random.rand(1,dimension) - 0.5))
                        rand_fill = np.zeros(2)
                    
                        for v in rand_value:        #to create array to add the random step size to the position
                            for count, w in enumerate(v):
                                
                                rand_fill[count] = w
                        
                        self.__agents[i] = list(map(int, rand_fill)) + self.__agents[i] #adds the randomize step size to the poistion of agent i

                    fitness[i] = function(self.__agents[i])  #after a move on agent[i] we calculate a new fitness value, to compare with other agents
                        
            self.__agents = np.clip(self.__agents, lb, ub)  #Checks that all agents are in the required bound, and change them to the highest or lowest if not (lb, or ub)
            
            self._points(self.__agents)
            
            Pbest = self.__agents[np.array(fitness).argmin()] #sets the Pbest of the iteration to the agent with the smallest value #[function(x) for x in self.__agents]
            #print("Pbest before we compare with Gbest", Pbest)
            if function(Pbest) < function(Gbest):
                
                Gbest = copy.deepcopy(Pbest)

            print(Pbest, "best solution for iteration: ", t, "With fitness value: ", function(Pbest), "with alpha being: ", alpha)    
        self._set_Gbest(Gbest)

    def __move(self, i, j, t, csi, psi, alpha, dimension, ub, lb):

        r = np.linalg.norm(self.__agents[i] - self.__agents[j]) #Gives the Forbenius norm, "same as cartesian distance"
        beta = csi / (1 + psi * (r ** 2)) #Approximative method to Changes the attractivness between the two agents depending on their distance r and mutual attraction cis
       
        rand_value_move = alpha * abs(ub-lb) * (np.random.rand(1,dimension) - 0.5)
        rand_fill_move = []
        for v in rand_value_move:
            for w in v:
                rand_fill_move.append(w)
        self.__agents[i] = self.__agents[i] + list(map(int, rand_fill_move)) + (beta * (self.__agents[i] - self.__agents[j]))
            