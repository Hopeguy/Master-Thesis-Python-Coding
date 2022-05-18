
from math import exp
import numpy as np
import intelligence
#From swarmpackagePY


class fa(intelligence.sw):
    """
    Firefly Algorithm
    """

    def __init__(self, n, function, lb1, ub1, lb2, ub2, dimension, iteration, csi=1, psi=1,
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

        ESS_capacity_start = np.random.uniform(lb1, ub1, n)
        ESS_power_start = np.random.uniform(lb2, ub2, n) 
        pre_prep_agents = []
        for count in range (0, n):
            pre_prep_agents.append([ESS_capacity_start[count], ESS_power_start[count]])
        self.__agents = pre_prep_agents   
            
        self._points(self.__agents)

        Pbest = self.__agents[np.array([function(x) for x in self.__agents]).argmin()]
        Gbest = Pbest

        for t in range(iteration):

            alpha = alpha1 + (alpha0 - alpha1) * exp(-t)

            for i in range(n):
                fitness = [function(x) for x in self.__agents]
                for j in range(n):
                    if fitness[i] > fitness[j]:
                        self.__move(i, j, t, csi, psi, alpha, dimension,
                                    norm0, norm1)
                    else:
                        self.__agents[i] += np.random.normal(norm0, norm1,
                                                             dimension)

            for count1, agent in enumerate (self.__agents):
                for count2, unit in enumerate(agent):
                    if count2 == 0:
                        self.__agents[count1][count2] = np.clip(unit, lb1, ub1)
                    if count2 == 1:
                        self.__agents[count1][count2] = np.clip(unit, lb2, ub2)
            self._points(self.__agents)

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)

    def __move(self, i, j, t, csi, psi, alpha, dimension, norm0, norm1):

        r = np.linalg.norm(self.__agents[i] - self.__agents[j])
        beta = csi / (1 + psi * r ** 2)

        self.__agents[i] = self.__agents[i] + beta * (                  #One thing is changed from the base version, it said self._agents[j] but it should be i instead of j
            self.__agents[i] - self.__agents[j]) + alpha * exp(-t) * \
                                                   np.random.normal(norm0, norm1, dimension)