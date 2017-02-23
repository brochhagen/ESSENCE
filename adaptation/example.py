from adaptation import run

'''
Run file: Compression-game with two players, prior convergence; preempter or plain
'''


##### Parameter specification #####
games = 100 #Amount of independent games
iterations_per_game = 50 #Number of interactions per independent game
confidence_upper_bound = 20 #Upper bound for dirichlet distributed sender beliefs
aug_value = 0.5 #Augmentation value for simple negative/positive reinforcement of state expectations
lam = 20 #Rationality parameter for linguistic behavior of players
context = 0.75 #Informativity of a context: P(s1)
adapt_strat = 'plain' #Sender inference strategy. Either 'plain' or 'preempter'
################################

#Example 1 with the values specified above:
run(games,iterations_per_game,confidence_upper_bound,aug_value,lam,context,adapt_strat)

#Example 2 with an uninformative context:
context = 0.5
run(games,iterations_per_game,confidence_upper_bound,aug_value,lam,context,adapt_strat)


#Decoment for example 3 and 4
##Example 3 with an informative context and a 'preemptive' learner
#context = 0.9
#adaptation_strategy = 'preempter'
#run(games,iterations_per_game,confidence_upper_bound,aug_value,lam,context,adapt_strat)
#
###Example 4 with an uninformative context and a 'preemptive' learner
#context = 0.5
#adaptation_strategy = 'preempter'
#run(games,iterations_per_game,confidence_upper_bound,aug_value,lam,context,adapt_strat)

