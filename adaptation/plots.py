import pandas as pd
import numpy as np 
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns


def plt_results(kind,g,i,a,r,l,c):
    print 'Loading data'
    df = pd.read_csv(r"./results/adaptation-%s-g%d-i%d-a%d-r%.2f-lambda%.2f-context%.2f.csv" % (kind,g,i,a,r,l,c))
    fig = plt.figure()
    group = df.loc[:,('iterations','alpha_upper','aug_r','context','phi','failures_iter','s_prior_initial','r_prior_initial','s_prior_over_iters','r_prior_over_iters')]
    
    X = np.arange(1,i+1,1)

    Y1 = [eval(group['s_prior_over_iters'][x]) for x in xrange(len(group['s_prior_over_iters']))] 
    Y1 = [sum(x) / len(x) for x in zip(*Y1)]

    Y2 = [eval(group['r_prior_over_iters'][x]) for x in xrange(len(group['r_prior_over_iters']))] 
    Y2 = [sum(x) / len(x) for x in zip(*Y2)]

    Y3 = [c for _ in xrange(i)]
    
    ax1 = fig.add_subplot(211)
    ax1.plot(X,Y1, marker='*', markersize=12,markevery=4)
    ax1.plot(X,Y2, marker='D', markersize=7,markevery=5)

    ax1.set_xlim(min(X),max(X))
    ax1.set_ylim(0,max(Y1+Y2)+0.025)
    ax1.set_ylabel("Probability of state 1",fontsize=14)
    ax1.set_xlabel('Iterations',fontsize=14)

    print 'Loading data'
    df = pd.read_csv(r"./results/adaptation-%s-g%d-i%d-a%d-r%.2f-lambda%.2f-context%.2f.csv" % (kind,g,i,a,r,l,c))
    group = df.loc[:,('iterations','alpha_upper','aug_r','context','phi','s_prior_final','r_prior_final','s_prior_over_priors_final')]
    
    X = np.arange(1,100) * 0.01
    Y_prior = [eval(group['s_prior_over_priors_final'][x]) for x in xrange(len(group['s_prior_over_priors_final']))]
    Y_prior = [sum(x) / len(x) for x in zip(*Y_prior)]
    
    X_sender = [group['s_prior_final'][x] for x in xrange(len(group['s_prior_final']))] #from game 0 THIS SHOULD BE AGGREGATED
    X_sender = sum(X_sender) / float(len(X_sender))
    X_sender = [X_sender for _ in xrange(len(X))]
    
    X_receiver = [group['r_prior_final'][x] for x in xrange(len(group['r_prior_final']))] #from game 0 THIS SHOULD BE AGGREGATED
    X_receiver = sum(X_receiver) / float(len(X_receiver))
    X_receiver = [X_receiver for _ in xrange(len(X))]
    
    X_context = [c for _ in xrange(len(X))]
    
    ax2 = fig.add_subplot(212) 
    ax2.plot(X,Y_prior)#,marker='d',markevery=10)
    ax2.axvline(x=c, color=sns.color_palette()[1],linestyle ='dashed', markevery=10)
    
    ax2.set_xlim(min(X),max(X))
    ax2.set_ylim(min(Y_prior), max(Y_prior))
    ax2.set_ylabel("Probability assigned to prior",fontsize=14)
    ax2.set_xlabel('Probability of state 1',fontsize=14)
    ax2.legend(["Beliefs about receiver's prior", "State frequency $P^*(s_1)$"],loc='best',fontsize=13)
    
    fig.tight_layout()
    plt.show()

