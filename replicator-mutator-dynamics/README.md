Readme (Replicator-Mutator Dynamics)
====================

This folder contains code to generate predictions about population level transmission and use of linguistic types (combinations of lexica and linguistic behavior). A pressure for communicative efficiency is exherted by the replicator dynamics, and a pressure for learnability by the mutator dynamics (corresponding to iterated Bayesian learning).

The code allows for variable specification of linguistic behavior (soft-maximization, best response, depth of mutual reasoning, ...), the kind of dynamics considered (replication only, mutation only, or replication and mutation), learning behavior (soft-maximization, posterior sampling/maximization), and data sparsity (sequence length, number of samples per parent type). An example that shows how the parameters may be modified is given in *example.py*

Details about the model's inner workings and theoretical background are found in:

> Thomas Brochhagen, Michael Franke & Robert van Rooij (2016): Learning biases may prevent lexicalization of pragmatic inferences: a case study combining iterated (Bayesian) learning and functional selection. Proceedings of the 38th Annual Conference of the Cognitive Science Society (CogSci 2016), pp. 2081-2086.

and, in more detail for the current type space, in forthcoming work:

> Thomas Brochhagen, Michael Franke & Robert van Rooij (manuscript): Tracing the cultural evolution of meaning at the semantics-pragmatics interface.

Contact for inquiries & manuscript: t.s.brochhagen@uva.nl



Example
---------------------

The file *example.py* is a self-contained example that runs 10 independent applications of the replication-mutator dynamics of 50 generations each. It outputs the incumbent type in a population of 432 types, as well as its proportion in the population.

Run as:
> python example.py
