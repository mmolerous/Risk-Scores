# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:22:16 2024

@author: crist
"""

from pyomo.environ import *
import random

random.seed(1000)

model = AbstractModel()

divided = 50

### PARAMS AND SETS
# Number of observations
model.n = Param(within=PositiveIntegers)
# Number of features
model.p = Param(within=PositiveIntegers)
# Budget
model.k = Param(within=PositiveIntegers)
# Set of observations
model.N = RangeSet(1,model.n)
# Set of features
model.P = RangeSet(1,model.p)
# Dataset
model.X = Param(model.N, model.P)
# Labels
model.Y = Param(model.N,within=Integers)

### DECISION VARIABLES
# w_leq[j]: weight linked to binarized feature X[j] <= t[j]
model.w_leq = Var(model.P,within=Integers,bounds=(-5,5))
# w_geq[j]: weight linked to binarized feature X[j] >= t[j]
model.w_geq = Var(model.P,within=Integers, bounds=(-5,5))
#,bounds=(-5,5))
# w0: intercept
model.w0 = Var(within=Integers, bounds=(-5*model.p,5*model.p)) # TBC
# alpha[j]: 1 if feature s is used
model.alpha = Var(model.P,within=Binary)
# t[j]: threshold for continuous feature j
model.t = Var(model.P,bounds=(0,1)) #TBC
# b[i,j]: 1 if x[i,j] <= t[j]
model.b = Var(model.N, model.P, within=Binary)
# omega_leq[i,j]: term added to the o.f. if x[i,j] <= t[j]
model.omega_leq = Var(model.N, model.P,within=Integers,bounds=(-5,5))
# omega_geq[i,j]: term added to the o.f. if x[i,j] >= t[j]
model.omega_geq = Var(model.N, model.P,within=Integers,bounds=(-5,5))

# Minimize the misclassification error
def error_(model):
     return sum(log(1+exp(-model.Y[i]*(sum(model.omega_leq[i,j] + model.omega_geq[i,j] for j in model.P) + model.w0)/divided)) for i in model.N)
model.error = Objective(rule=error_, sense=minimize)

# (21). Maximum number of features used
def budget_(model):
    return sum(model.alpha[j] for j in model.P) <= model.k
model.budget = Constraint(rule=budget_)

# (22).  alpha and w_leq constraints
def alphaw1leq_(model,j):
    return model.w_leq[j] - 5*model.alpha[j] <=  0
model.alphaw1leq = Constraint(model.P, rule=alphaw1leq_)

def alphaw2leq_(model,j):
    return - model.w_leq[j] - 5*model.alpha[j] <=  0
model.alphaw2leq = Constraint(model.P, rule=alphaw2leq_)

# (23).  alpha and w_geq constraints
def alphaw1geq_(model,j):
    return model.w_geq[j] - 5*model.alpha[j] <=  0
model.alphaw1geq = Constraint(model.P, rule=alphaw1geq_)

def alphaw2geq_(model,j):
    return - model.w_geq[j] - 5*model.alpha[j] <=  0
model.alphaw2geq = Constraint(model.P, rule=alphaw2geq_)

# (24). Constraints for continuous discretization. TBC
def continuous1_(model,i,j):
    return -model.b[i,j] + model.t[j] - model.X[i,j] <=  0
model.continuous1 = Constraint(model.N, model.P, rule=continuous1_)

def continuous2_(model,i,j):
    return model.X[i,j] - model.t[j] - (1-model.b[i,j])  <=  0
model.continuous2 = Constraint(model.N, model.P, rule=continuous2_)

# (25) 
def omega1leq_(model,i,j):
    return  -5*model.b[i,j] + model.omega_leq[i,j] <=  0
model.omega1leq = Constraint(model.N, model.P, rule=omega1leq_)

# (26) 
def omega2leq_(model,i,j):
#    return  -model.w_leq[j] + model.omega_leq[i,j] <=  0
    return  -model.w_leq[j] + 5*model.b[i,j] -5 + model.omega_leq[i,j] <=  0
model.omega2leq = Constraint(model.N, model.P, rule=omega2leq_)

# (27) 
def omega3leq_(model,i,j):
    return  -5*model.b[i,j] - model.omega_leq[i,j] <=  0
model.omega3leq = Constraint(model.N, model.P, rule=omega3leq_)

# (28) 
def omega4leq_(model,i,j):
    return  5*model.b[i,j] + model.w_leq[j] -5 - model.omega_leq[i,j] <=  0
model.omega4leq = Constraint(model.N, model.P, rule=omega4leq_)

# (29) 
def omega1geq_(model,i,j):
    return  -5 + 5*model.b[i,j] + model.omega_geq[i,j] <=  0
model.omega1geq = Constraint(model.N, model.P, rule=omega1geq_)

# (30) 
def omega2geq_(model,i,j):
    # return  -model.w_geq[j] + model.omega_geq[i,j] <=  0
    return  -model.w_geq[j] - 5*model.b[i,j] + model.omega_geq[i,j] <=  0
model.omega2geq = Constraint(model.N, model.P, rule=omega2geq_)

# (31) 
def omega3geq_(model,i,j):
    return  -5 + 5*model.b[i,j] - model.omega_geq[i,j] <=  0
model.omega3geq = Constraint(model.N, model.P, rule=omega3geq_)

# (32) 
def omega4geq_(model,i,j):
    return  -5*model.b[i,j] + model.w_geq[j] - model.omega_geq[i,j] <=  0
model.omega4geq = Constraint(model.N, model.P, rule=omega4geq_)

