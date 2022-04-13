import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

n = 20       #number of agents (fireflies)
fitness_function = tf.ackley_function      #fitness function to be used
lb = -5  #lower bound of search space (plot axis)
ub = 5 #Higher bound of search space (plot axis)
dimensions = 2 
iteration = 20   #number of iterations the algorithm will run



csi = 1     #mutal attraction value
psi = 1     #Light absoprtion coefficent
alpha0 = 1   #initial value of the free randomization parameter alpha
alpha1 = 0.1 #final value of the free randomization parameter alpha
norm0 = 0  #first parameter for a normal (Gaussian) distribution 
norm1 = 0.1  #second parameter for a normal (Gaussian) distribution #as we are looking at ints these are not normal gassuian


alh = SwarmPackagePy.fa(n = n, function = fitness_function, lb = lb, ub = ub, dimension = dimensions,
                         iteration = iteration, csi = csi, psi =  psi, alpha0 = alpha0,
                        alpha1 = alpha1, norm0 = norm0, norm1 = norm1)

animation(alh.get_agents(), fitness_function, lb, ub)
animation3D(alh.get_agents(), fitness_function, lb, ub)
print(alh.get_Gbest())
print()