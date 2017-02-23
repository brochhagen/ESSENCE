##### Main file to run dynamics
from rmd import run_dynamics

##### Parameters & setup #####
alpha = 1 # rate to control difference between semantic and pragmatic violations
lamb = 30 # soft-max parameter
seq_length = 10  # length of observation sequences
samples = 250 #amount of k-length samples for each production type
learning_parameter = 10 #prob-matching = 1, increments approach MAP
g = 50 #number of generations per simulation run
r = 10 #number of independent simulation runs
dynamics = 'rmd' #kind is the type of dynamics, either 'rmd', 'm' or 'r'

states = 3 #number of states
messages = 3 #number of messages
me = False #mutual exclusivity
###############################


#Example run with the parameter values specified above#
run_dynamics(alpha,lamb,seq_length,samples,g,r,states,messages,learning_parameter,dynamics,me) 
