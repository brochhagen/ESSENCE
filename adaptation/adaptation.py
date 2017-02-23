from __future__ import division
from Player import Player
import random
import numpy as np
import sys
import csv
from math import ceil
'''
Main file: Compression-game with two players, prior convergence; preempter or plain
'''

def cost(player,m): #cost of a message for a player
    return player.costVector[0][player.signals.index(m)]

def utilityS(t,m,a,sender): #sender utility for <t,a> - cost(m)
    if t == a:
        return 1 - cost(sender,m)
    else:
        return 0 - cost(sender,m)

def utilityR(t,a): #receiver utility for <t,a>
    if t == a:
        return 1 
    else: return 0 

def updateRewards(player, state, util,augvalue): #prior update of player for state given util with augvalue
    if util > 0:
        player.accumulatedRewards[state] += augvalue
    elif (player.accumulatedRewards[state] - augvalue) > 0:
        player.accumulatedRewards[state] -= augvalue
    else: #failsafe to not get to or below 0
        player.accumulatedRewards[state] = 0.00001

def init_prior_over_priors(k,priors,player_prior): #initialization of dirichlet distributed sender beliefs about receiver expectations
    weight_s1 = k * player_prior + 1
    weight_s2 = k * (1 - player_prior) + 1
    alpha = (weight_s1,weight_s2)
    #Approximating Dir as PMF through sampling
    samples = np.random.dirichlet(alpha, 20000)
    out = np.ones(99)
    for x in xrange(len(samples)):
        s_1 = samples[x][0] * 100 #P(s_1) * 100 to discretize and avoid rounding problems
        idx = int(ceil(s_1))
        if not(idx == 100 or idx == 0):
            idx = idx - 1 #ints to python idx
            out[idx] +=1
    return out / np.sum(out)

def updatePrOverPr(player,state,message,util, kind='plain'): #Update of sender beliefs over receiver expectations either as 'plain' learner or 'preemptive' learner
    if util > 0:
        if state == 't1':
            s = 0 #w(s_i) == s_i iff util > 0
            q = 0
        else:
            s = 1
            q = 0
    if util < 0:
        if state == 't1':
            s = 1
            q = 1
        else:
            s = 0
            q = 1
    post = np.zeros(len(player.inter_priors))
    for idx in xrange(len(player.inter_priors)):
        m_idx = player.signals.index(message)
        pr_interpretation = player.fict_receiver(player.inter_priors[idx])[m_idx,s] #probability of producing evinced behavior
        if kind == 'plain':
            pr_given_outcome = 1
        elif kind == 'preempter':
            pr_given_outcome = player.inter_priors[idx][q] #probability of subjective expectations given the interaction's outcome
        post[idx] = pr_interpretation * player.pr_over_pr[idx] * pr_given_outcome
    player.pr_over_pr = post / np.sum(post)
   
def euS(player1,player2,c): 
    """Expected utility of a sender p1 against receiver p2"""
    context = np.array([c,1-c])
    sender_cost = np.ones(len(player1.signals)) - player1.costVector
    utils = (player1.sigma() * np.transpose(player2.rho())) * sender_cost
    rel_utils = np.sum(np.transpose(utils) * context)
    return rel_utils

def euR(player1,player2,c):
    """Expected utility of a receiver p1 against sender p2"""
    context = np.array([c,1-c])
    utils = (player2.sigma() * np.transpose(player1.rho()))
    rel_utils = np.sum(np.transpose(utils) * context)
    return rel_utils

def EU(player1,player2,c):
    """Symmetrized expected utility of two players"""
    return (euS(player1,player2,c) + euR(player2,player1,c)) / 2

def pr_star(player): #aux function for analysis
        prPrime = np.zeros(len(player.types))
        for idx_s in xrange(len(prPrime)):
            for idx_pr in xrange(len(player.inter_priors)):
                prPrime[idx_s] += player.pr_over_pr[idx_pr] * player.inter_priors[idx_pr][idx_s]
        return prPrime

def run(g,i,conf,r,lam,context,adapt_strat): #main function to run g-games of i-iterations with an upper-bound in dirichlet distributed beliefs of conf, an augmentation value of r, rationality parameter lam, a context, and a (sender) adaptation strategy
    lexicon = np.array([[1,0,1],[0,1,1]])
    costVector = np.array([[0.4,0.4,0.1]])
    poss_priors = [[x * 0.01,(100-x)*0.01] for x in xrange(1,100)]
    games = g
    iterations = i
    augvalue = r
    typeDistribution = [context,(1-context)]    
    
    f = csv.writer(open('./results/adaptation-%s-g%d-i%d-a%d-r%.2f-lambda%.2f-context%.2f.csv' %(adapt_strat, g,i,conf,r,lam,context),'wb')) #file to store mean results

    f.writerow(["run_ID", "iterations", "alpha_upper","aug_r", "lambda", "context", "kind", "s_prior_initial", "s_alpha_initial","s_prior_over_priors_initial", "r_prior_initial","eu_initial", "euS_initial","euR_initial","s_prior_final","s_prior_over_priors_final", "r_prior_final","eu_final","euS_final","euR_final","failures_iter", "s_prior_over_iters","r_prior_over_iters"])

    for games in xrange(g):

        s_prior = random.randint(1,99)*0.01 #random initial prior for sender 
        r_prior = random.randint(1,99)*0.01 #random initial prior for receiver
        s_conf =  random.randint(0,conf) #k used as weight for Dirichlet
        s_pr_over_pr = init_prior_over_priors(s_conf,poss_priors,s_prior)
        
        sender = Player(lexicon,costVector,lam,s_prior,poss_priors,s_pr_over_pr)
        receiver = Player(lexicon,costVector,lam,r_prior,poss_priors,s_pr_over_pr) 
        
        euS_init = euS(sender,receiver,context)
        euR_init = euR(receiver,sender,context)
        eu_init = EU(sender,receiver,context)

        print '### Game %d ###' %games

        fail = []
        s_prior_over_iters = []
        r_prior_over_iters = []

        for iters in xrange(i):

            state = ['t1','t2'][int(np.random.choice(2,1,p=typeDistribution))]
            message  = sender.selectSignal(state)
            act = receiver.selectAct(message)
            uS = utilityS(state,message,act,sender)
            uR = utilityR(state,act)

            if uR <= 0:
                fail.append(iters)

            s_prior_over_iters.append(sender.priorAC()[0])
            r_prior_over_iters.append(receiver.priorAC()[0])

            updateRewards(sender,state,uS,augvalue)
            updateRewards(receiver,act,uR,augvalue)
            updatePrOverPr(sender,state,message,uS,kind=adapt_strat) #kinds: plain, preempter

        euS_final = euS(sender,receiver,context)
        euR_final = euR(receiver,sender,context)
        eu_final = EU(sender,receiver,context)
        print '#final prior sender:', sender.priorAC()
        prPrime = pr_star(sender)
        print 'pr^*:', prPrime
        print '#final prior receiver:', receiver.priorAC()
        print '#final EU sender, receiver, and symmetrized:', euS_final, euR_final, eu_final
        print '#failures in iterations #: ', fail
        f.writerow([str(games), str(iterations), str(conf), str(augvalue), str(lam), str(context), str(adapt_strat), str(s_prior), str(s_conf), repr(s_pr_over_pr), str(r_prior), str(eu_init), str(euS_init), str(euR_init), str(sender.priorAC()[0]), repr(sender.pr_over_pr), str(receiver.priorAC()[0]), str(eu_final), str(euS_final), str(euR_final), str(fail),str(s_prior_over_iters),str(r_prior_over_iters)])
