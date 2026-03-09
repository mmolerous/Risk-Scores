Repository Risk-Scores contains the code for the approach proposed by C. Molero-Río 
and C. D'Ambrosio in "Optimal risk scores for continuous predictors" to be appeared
in Lecture Notes of Computer Science as part of the proceedings of LOD conference.

All the files in this repository can be run in Python 3.11, and require BONMIN 1.8.9 
and IPOPT 3.12.13 solvers.

-> MAIN FILE:

	* 'model_new.py': formulation of the problem using Pyomo modelling language.

-> AUXILIARY FILES to obtain the results in the paper:

	* 'run_all.py': run model_new.py over the synthetic datasets with different parameters 
         configuration for both BONMIN 1.8.9 and IPOPT 3.12.13 solvers. 

	* 'retrieving_ipopt_neq.py': run Algorithm 1 over the results obtained with IPOPT 3.12.13
         solver.
