# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:51:57 2024

@author: crist
"""
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

# Experiment 1
for which_solver in ["bonmin","ipopt"]:
    TL = 1000
    
    for p in [1,2,3,4,5]:
        np.random.seed(0)
        n = 200
        #p = 1
        k0 = p
        
        a = 0
        b = 1
        
        w0 = 0
        
        if p==1:    
            w_leq = np.array([2]) #p1
            w_geq = np.array([0]) #p1
            t = np.array([0.4]) #p1
        elif p ==2:    
            w_leq = np.array([2,0]) #p2
            w_geq = np.array([0,-5]) #p2
            t = np.array([0.4,0.2]) #p2
        elif p ==3:
            w_leq = np.array([2,0,3]) #p3
            w_geq = np.array([0,-5,0]) #p3
            t = np.array([0.4,0.2,0.7]) #p3
        elif p == 4:
            w_leq = np.array([2,0,3,-1]) #p4
            w_geq = np.array([0,-5,0,0]) #p4
            t = np.array([0.4,0.2,0.7,0.8]) #p4
        elif p ==5:
            w_leq = np.array([2,0,3,-1,0]) #p5
            w_geq = np.array([0,-5,0,0,4]) #p5
            t = np.array([0.4,0.2,0.7,0.8,0.1]) #p5
        x = np.zeros((n, p))
        for j in range(p):
            x[:,j] = np.random.uniform(a,b,n)
        
        y = np.ones(n)
        is_leq = np.zeros((n,p))
        is_geq = np.zeros((n,p))
        out = []
        proba = []
        for i in range(n):
            for j in range(p):
                is_leq[i,j] = int(x[i,j] <= t[j])
                is_geq[i,j] = int(x[i,j] >= t[j])
                out.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0))==0.5)
                proba.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)))
            if np.round(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)),0) == 0:
                y[i] = -1
        
        xy = np.zeros((n,p+1))
        xy[:,0:p] = x[:,0:p]
        xy[:,p] = y
        
        path = "C:/Users/crist/Desktop/PARIS/RuleSets"  
        os.chdir(path)
        
        df = pd.DataFrame(xy)
        listvars = []
        for j in range(p):
            listvars.append('v'+str(j+1))
        listvars.append('Class')
        df.columns = listvars
        fea = df.columns[0:(len(df.columns)-1)]
        X = df[fea] # Features
        y = df.Class # Target variable
        
        # Normalization to the 0-1 interval
        minimum = {}
        maximum = {}
        for j in range(len(X.columns)):
            minimum[j] = min(X[X.columns[j]])
            maximum[j] = max(X[X.columns[j]])
            for i in range(0,len(X)):
                X.iloc[i,j] = (X.iloc[i,j]-minimum[j])/(maximum[j]-minimum[j])
        
        X0={}
        for i in range(0,len(X)):
            for j in range(len(X.columns)):
                X0[i+1,j+1]=X.iloc[i][j]
                
        y0={}
        for i in range(0,len(y)):
            if y[i]==0:
                y[i] += -1
            y0[i+1]=list(y)[i]
        
        data = {None: dict(
                n = {None : len(X)},
                p = {None : len(X.columns)},
                k = {None : k0},
                X = X0,
                Y = y0)
                }
        
        import time
        
        from model_new import model
        
        if which_solver == "ipopt":
            pathsolver = "C:/coin/ipopt.exe"
            opt = SolverFactory("ipopt",executable=pathsolver,time_limit=TL)
        else:
            pathsolver = "C:/coin/bonmin.exe"
            opt = SolverFactory("bonmin",executable=pathsolver)
            opt.set_options('bonmin.time_limit=' +str(TL) + ' bonmin.algorithm=B-Hyb')
        
        instance = model.create_instance(data) 
        
        # This is code for giving an initial solution:
        #instance.w0 =  w0
        #for i in range(0,len(X)):
        #    for j in range(len(X.columns)):
        #        instance.b[i+1,j+1] = is_leq[i,j]
        #        instance.omega_leq[i+1,j+1] = is_leq[i,j]*w_leq[j]
        #        instance.omega_geq[i+1,j+1] = is_geq[i,j]*w_geq[j]      
        #for j in range(len(X.columns)):
        #    instance.w_leq[j+1] = w_leq[j]
        #    instance.w_geq[j+1] = w_geq[j]
        #    instance.t[j+1] = t[j]
        #    instance.alpha[j+1] = 1
        
        import sys
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f
            
        t0 = time.time()
        results = opt.solve(instance,tee=True,keepfiles=True, logfile = path + "/results_auc/" +  which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + "_log.log")
        t1 = time.time()
        T = t1-t0
        
        try:
            instance.solutions.load_from(results)
            print("\n")
            for j in range(len(X.columns)):
                print('alpha_' + str(j+1), '=', instance.alpha[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_leq_' + str(j+1), '=', instance.w_leq[j+1].value, '*', X.columns[j],'<=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_geq_' + str(j+1), '=', instance.w_geq[j+1].value, '*', X.columns[j],'>=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('t_' + str(j+1), '=', instance.t[j+1].value)
            print("\n")
            print('w_0 =', instance.w0.value)
            print("\n")
            
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('b_' + str(i+1) + '_' + str(j+1) + '=', instance.b[j+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_leq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_leq[i+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_geq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_geq[i+1,j+1].value)
            
            print("\n")
            
            
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)
            print('time =', T)
            objective_value_solver = instance.error()
            print('objective_value_solver', objective_value_solver)
            
            ###############
            # ACC OMEGA
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_alpha = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            r_b = np.zeros((n,p))
            r_omega_leq = np.zeros((n,p))
            r_omega_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                r_alpha[j] = instance.alpha[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
                    r_b[i,j] = instance.b[i+1,j+1].value
                    r_omega_leq[i,j] = instance.omega_leq[i+1,j+1].value
                    r_omega_geq[i,j] = instance.omega_geq[i+1,j+1].value
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq[i,j] + r_omega_geq[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(y==finalpred)/n 
            print('acc_omega', acc_omega)
            
            auc_omega = roc_auc_score(y, pred_omega)
            print('auc_omega', auc_omega)
            
            ###############
            
            # ###############
            # ACC WC
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
            pred_wc = np.zeros(n)
            for i in range(n):
                pred_wc[i] = pred_wc[i] + r_w0
                for j in range(p):
                    pred_wc[i] = pred_wc[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                pred_wc[i] = 1/(1+np.exp(-pred_wc[i]))
            
            finalpred = np.round(pred_wc,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
                   
            acc_wc = sum(y==finalpred)/n
            print('acc_wc', acc_wc)
            
            auc_wc = roc_auc_score(y, pred_wc)
            print('auc_wc', auc_wc)
            # ###############
            
            print("\n")
            
            # ACC train
            pred = np.zeros(n)
            for i in range(n):
                pred[i] = pred[i] + w0
                for j in range(p):
            #        pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*is_geq[i,j]
                    pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*(1-is_leq[i,j])
                pred[i] = 1/(1+np.exp(-pred[i]))
            
            finalpred = np.round(pred,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            acc_OptimalSolution = sum(y==finalpred)/n
            print('acc_OptimalSolution', acc_OptimalSolution)
            
            auc_OptimalSolution = roc_auc_score(y, pred)
            print('auc_OptimalSolution', auc_OptimalSolution)
            
            # Objective value
            test = 0
            for i in range(n):
                term_leq = 1*(x[i,:] <= t)
                #term_leq = 1*(X.iloc[i,:] <= t)
                #print(np.sum(w_leq*term_leq))
                term_geq = 1*(x[i,:] >= t)
                #print(sum(term_leq+term_geq))
                #term_geq = 1*(x[i,:] >= t)
                #print(np.sum(w_geq*term_geq))
               # test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*term_leq) + np.sum(w_geq*term_geq) + w0)))
                test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*is_leq[i,:]) + np.sum(w_geq*is_geq[i,:]) + w0)))
            
            print('objective_value_OptimalSolution', test)
            
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0"] = r_w0
            shelf["r_w_leq"] = r_w_leq
            shelf["r_w_geq"] = r_w_geq
            shelf["r_t"] = r_t
            shelf["r_alpha"] = r_alpha
            shelf["r_b"] = r_b
            shelf["r_check_leq"] = r_check_leq
            shelf["r_check_geq"] = r_check_geq
            shelf["r_beta_leq"] = r_omega_leq
            shelf["r_beta_geq"] = r_omega_geq
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["w0"] = w0
            shelf["w_leq"] = w_leq
            shelf["w_geq"] = w_geq
            shelf["t"] = t 
            shelf["X"] = X
            shelf["y"] = y 
            shelf["X0"] = X0 
            shelf["T"] = T 
            shelf["objective_value_solver"] = objective_value_solver
            shelf["pred_omega"] = pred_omega
            shelf["acc_omega"] = acc_omega
            shelf["auc_omega"] = auc_omega
            shelf["pred_wc"] = pred_wc
            shelf["acc_wc"] = acc_wc  
            shelf["auc_wc"] = auc_wc  
            shelf["pred_OptimalSolution)"] = pred
            shelf["acc_OptimalSolution"] = acc_OptimalSolution
            shelf["objective_value_OptimalSolution"] = test
            shelf["auc_OptimalSolution"] = auc_OptimalSolution
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()
        
# Experiment 2
for which_solver in ["bonmin","ipopt"]:
    TL = 1000
    
    for k0 in [1,2,3,4]:
        np.random.seed(0)
        n = 200
        p = 5
        #k0 = p
        
        a = 0
        b = 1
        
        w0 = 0
        
        if p==1:    
            w_leq = np.array([2]) #p1
            w_geq = np.array([0]) #p1
            t = np.array([0.4]) #p1
        elif p ==2:    
            w_leq = np.array([2,0]) #p2
            w_geq = np.array([0,-5]) #p2
            t = np.array([0.4,0.2]) #p2
        elif p ==3:
            w_leq = np.array([2,0,3]) #p3
            w_geq = np.array([0,-5,0]) #p3
            t = np.array([0.4,0.2,0.7]) #p3
        elif p == 4:
            w_leq = np.array([2,0,3,-1]) #p4
            w_geq = np.array([0,-5,0,0]) #p4
            t = np.array([0.4,0.2,0.7,0.8]) #p4
        elif p ==5:
            w_leq = np.array([2,0,3,-1,0]) #p5
            w_geq = np.array([0,-5,0,0,4]) #p5
            t = np.array([0.4,0.2,0.7,0.8,0.1]) #p5
        x = np.zeros((n, p))
        for j in range(p):
            x[:,j] = np.random.uniform(a,b,n)
        
        y = np.ones(n)
        is_leq = np.zeros((n,p))
        is_geq = np.zeros((n,p))
        out = []
        proba = []
        for i in range(n):
            for j in range(p):
                is_leq[i,j] = int(x[i,j] <= t[j])
                is_geq[i,j] = int(x[i,j] >= t[j])
                out.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0))==0.5)
                proba.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)))
            if np.round(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)),0) == 0:
                y[i] = -1
        
        xy = np.zeros((n,p+1))
        xy[:,0:p] = x[:,0:p]
        xy[:,p] = y
        
        path = "C:/Users/crist/Desktop/PARIS/RuleSets"  
        os.chdir(path)
        
        df = pd.DataFrame(xy)
        listvars = []
        for j in range(p):
            listvars.append('v'+str(j+1))
        listvars.append('Class')
        df.columns = listvars
        fea = df.columns[0:(len(df.columns)-1)]
        X = df[fea] # Features
        y = df.Class # Target variable
        
        # Normalization to the 0-1 interval
        minimum = {}
        maximum = {}
        for j in range(len(X.columns)):
            minimum[j] = min(X[X.columns[j]])
            maximum[j] = max(X[X.columns[j]])
            for i in range(0,len(X)):
                X.iloc[i,j] = (X.iloc[i,j]-minimum[j])/(maximum[j]-minimum[j])
        
        X0={}
        for i in range(0,len(X)):
            for j in range(len(X.columns)):
                X0[i+1,j+1]=X.iloc[i][j]
                
        y0={}
        for i in range(0,len(y)):
            if y[i]==0:
                y[i] += -1
            y0[i+1]=list(y)[i]
        
        data = {None: dict(
                n = {None : len(X)},
                p = {None : len(X.columns)},
                k = {None : k0},
                X = X0,
                Y = y0)
                }
        
        import time
        
        from model_new import model
        
        if which_solver == "ipopt":
            pathsolver = "C:/coin/ipopt.exe"
            opt = SolverFactory("ipopt",executable=pathsolver,time_limit=TL)
        else:
            pathsolver = "C:/coin/bonmin.exe"
            opt = SolverFactory("bonmin",executable=pathsolver)
            opt.set_options('bonmin.time_limit=' +str(TL) + ' bonmin.algorithm=B-Hyb')
        
        instance = model.create_instance(data) 
        
        # This is code for giving an initial solution:
        #instance.w0 =  w0
        #for i in range(0,len(X)):
        #    for j in range(len(X.columns)):
        #        instance.b[i+1,j+1] = is_leq[i,j]
        #        instance.omega_leq[i+1,j+1] = is_leq[i,j]*w_leq[j]
        #        instance.omega_geq[i+1,j+1] = is_geq[i,j]*w_geq[j]      
        #for j in range(len(X.columns)):
        #    instance.w_leq[j+1] = w_leq[j]
        #    instance.w_geq[j+1] = w_geq[j]
        #    instance.t[j+1] = t[j]
        #    instance.alpha[j+1] = 1
        
        import sys
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f
            
        t0 = time.time()
        results = opt.solve(instance,tee=True,keepfiles=True, logfile = path + "/results_auc/" +  which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + "_log.log")
        t1 = time.time()
        T = t1-t0
        
        try:
            instance.solutions.load_from(results)
            print("\n")
            for j in range(len(X.columns)):
                print('alpha_' + str(j+1), '=', instance.alpha[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_leq_' + str(j+1), '=', instance.w_leq[j+1].value, '*', X.columns[j],'<=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_geq_' + str(j+1), '=', instance.w_geq[j+1].value, '*', X.columns[j],'>=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('t_' + str(j+1), '=', instance.t[j+1].value)
            print("\n")
            print('w_0 =', instance.w0.value)
            print("\n")
            
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('b_' + str(i+1) + '_' + str(j+1) + '=', instance.b[j+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_leq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_leq[i+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_geq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_geq[i+1,j+1].value)
            
            print("\n")
            
            
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)
            print('time =', T)
            objective_value_solver = instance.error()
            print('objective_value_solver', objective_value_solver)
            
            ###############
            # ACC OMEGA
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_alpha = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            r_b = np.zeros((n,p))
            r_omega_leq = np.zeros((n,p))
            r_omega_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                r_alpha[j] = instance.alpha[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
                    r_b[i,j] = instance.b[i+1,j+1].value
                    r_omega_leq[i,j] = instance.omega_leq[i+1,j+1].value
                    r_omega_geq[i,j] = instance.omega_geq[i+1,j+1].value
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq[i,j] + r_omega_geq[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(y==finalpred)/n 
            print('acc_omega', acc_omega)
            
            auc_omega = roc_auc_score(y, pred_omega)
            print('auc_omega', auc_omega)
            
            ###############
            
            # ###############
            # ACC WC
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
            pred_wc = np.zeros(n)
            for i in range(n):
                pred_wc[i] = pred_wc[i] + r_w0
                for j in range(p):
                    pred_wc[i] = pred_wc[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                pred_wc[i] = 1/(1+np.exp(-pred_wc[i]))
            
            finalpred = np.round(pred_wc,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
                   
            acc_wc = sum(y==finalpred)/n
            print('acc_wc', acc_wc)
            
            auc_wc = roc_auc_score(y, pred_wc)
            print('auc_wc', auc_wc)
            # ###############
            
            print("\n")
            
            # ACC train
            pred = np.zeros(n)
            for i in range(n):
                pred[i] = pred[i] + w0
                for j in range(p):
            #        pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*is_geq[i,j]
                    pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*(1-is_leq[i,j])
                pred[i] = 1/(1+np.exp(-pred[i]))
            
            finalpred = np.round(pred,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            acc_OptimalSolution = sum(y==finalpred)/n
            print('acc_OptimalSolution', acc_OptimalSolution)
            
            auc_OptimalSolution = roc_auc_score(y, pred)
            print('auc_OptimalSolution', auc_OptimalSolution)
            
            # Objective value
            test = 0
            for i in range(n):
                term_leq = 1*(x[i,:] <= t)
                #term_leq = 1*(X.iloc[i,:] <= t)
                #print(np.sum(w_leq*term_leq))
                term_geq = 1*(x[i,:] >= t)
                #print(sum(term_leq+term_geq))
                #term_geq = 1*(x[i,:] >= t)
                #print(np.sum(w_geq*term_geq))
               # test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*term_leq) + np.sum(w_geq*term_geq) + w0)))
                test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*is_leq[i,:]) + np.sum(w_geq*is_geq[i,:]) + w0)))
            
            print('objective_value_OptimalSolution', test)
            
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0"] = r_w0
            shelf["r_w_leq"] = r_w_leq
            shelf["r_w_geq"] = r_w_geq
            shelf["r_t"] = r_t
            shelf["r_alpha"] = r_alpha
            shelf["r_b"] = r_b
            shelf["r_check_leq"] = r_check_leq
            shelf["r_check_geq"] = r_check_geq
            shelf["r_beta_leq"] = r_omega_leq
            shelf["r_beta_geq"] = r_omega_geq
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["w0"] = w0
            shelf["w_leq"] = w_leq
            shelf["w_geq"] = w_geq
            shelf["t"] = t 
            shelf["X"] = X
            shelf["y"] = y 
            shelf["X0"] = X0 
            shelf["T"] = T 
            shelf["objective_value_solver"] = objective_value_solver
            shelf["pred_omega"] = pred_omega
            shelf["acc_omega"] = acc_omega
            shelf["auc_omega"] = auc_omega
            shelf["pred_wc"] = pred_wc
            shelf["acc_wc"] = acc_wc  
            shelf["auc_wc"] = auc_wc  
            shelf["pred_OptimalSolution)"] = pred
            shelf["acc_OptimalSolution"] = acc_OptimalSolution
            shelf["objective_value_OptimalSolution"] = test
            shelf["auc_OptimalSolution"] = auc_OptimalSolution
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()
        
# Experiment 3
for which_solver in ["bonmin","ipopt"]:
    TL = 1000
    
    for n in [1000,5000,10000,15000,20000]:
    #[1000,2000,3000,4000,5000,10000,11000,12000,13000,14000,15000,20000]:
        np.random.seed(0)
        #n = 200
        p = 1
        k0 = 1
        
        a = 0
        b = 1
        
        w0 = 0
        
        if p==1:    
            w_leq = np.array([2]) #p1
            w_geq = np.array([0]) #p1
            t = np.array([0.4]) #p1
        elif p ==2:    
            w_leq = np.array([2,0]) #p2
            w_geq = np.array([0,-5]) #p2
            t = np.array([0.4,0.2]) #p2
        elif p ==3:
            w_leq = np.array([2,0,3]) #p3
            w_geq = np.array([0,-5,0]) #p3
            t = np.array([0.4,0.2,0.7]) #p3
        elif p == 4:
            w_leq = np.array([2,0,3,-1]) #p4
            w_geq = np.array([0,-5,0,0]) #p4
            t = np.array([0.4,0.2,0.7,0.8]) #p4
        elif p ==5:
            w_leq = np.array([2,0,3,-1,0]) #p5
            w_geq = np.array([0,-5,0,0,4]) #p5
            t = np.array([0.4,0.2,0.7,0.8,0.1]) #p5
        x = np.zeros((n, p))
        for j in range(p):
            x[:,j] = np.random.uniform(a,b,n)
        
        y = np.ones(n)
        is_leq = np.zeros((n,p))
        is_geq = np.zeros((n,p))
        out = []
        proba = []
        for i in range(n):
            for j in range(p):
                is_leq[i,j] = int(x[i,j] <= t[j])
                is_geq[i,j] = int(x[i,j] >= t[j])
                out.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0))==0.5)
                proba.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)))
            if np.round(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)),0) == 0:
                y[i] = -1
        
        xy = np.zeros((n,p+1))
        xy[:,0:p] = x[:,0:p]
        xy[:,p] = y
        
        path = "C:/Users/crist/Desktop/PARIS/RuleSets"  
        os.chdir(path)
        
        df = pd.DataFrame(xy)
        listvars = []
        for j in range(p):
            listvars.append('v'+str(j+1))
        listvars.append('Class')
        df.columns = listvars
        fea = df.columns[0:(len(df.columns)-1)]
        X = df[fea] # Features
        y = df.Class # Target variable
        
        # Normalization to the 0-1 interval
        minimum = {}
        maximum = {}
        for j in range(len(X.columns)):
            minimum[j] = min(X[X.columns[j]])
            maximum[j] = max(X[X.columns[j]])
            for i in range(0,len(X)):
                X.iloc[i,j] = (X.iloc[i,j]-minimum[j])/(maximum[j]-minimum[j])
        
        X0={}
        for i in range(0,len(X)):
            for j in range(len(X.columns)):
                X0[i+1,j+1]=X.iloc[i][j]
                
        y0={}
        for i in range(0,len(y)):
            if y[i]==0:
                y[i] += -1
            y0[i+1]=list(y)[i]
        
        data = {None: dict(
                n = {None : len(X)},
                p = {None : len(X.columns)},
                k = {None : k0},
                X = X0,
                Y = y0)
                }
        
        import time
        
        from model_new import model
        
        if which_solver == "ipopt":
            pathsolver = "C:/coin/ipopt.exe"
            opt = SolverFactory("ipopt",executable=pathsolver,time_limit=TL)
        else:
            pathsolver = "C:/coin/bonmin.exe"
            opt = SolverFactory("bonmin",executable=pathsolver)
            opt.set_options('bonmin.time_limit=' +str(TL) + ' bonmin.algorithm=B-Hyb')
        
        instance = model.create_instance(data) 
        
        # This is code for giving an initial solution:
        #instance.w0 =  w0
        #for i in range(0,len(X)):
        #    for j in range(len(X.columns)):
        #        instance.b[i+1,j+1] = is_leq[i,j]
        #        instance.omega_leq[i+1,j+1] = is_leq[i,j]*w_leq[j]
        #        instance.omega_geq[i+1,j+1] = is_geq[i,j]*w_geq[j]      
        #for j in range(len(X.columns)):
        #    instance.w_leq[j+1] = w_leq[j]
        #    instance.w_geq[j+1] = w_geq[j]
        #    instance.t[j+1] = t[j]
        #    instance.alpha[j+1] = 1
        
        import sys
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f
            
        t0 = time.time()
        results = opt.solve(instance,tee=True,keepfiles=True, logfile = path + "/results_auc/" +  which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + "_log.log")
        t1 = time.time()
        T = t1-t0
        
        try:
            instance.solutions.load_from(results)
            print("\n")
            for j in range(len(X.columns)):
                print('alpha_' + str(j+1), '=', instance.alpha[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_leq_' + str(j+1), '=', instance.w_leq[j+1].value, '*', X.columns[j],'<=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_geq_' + str(j+1), '=', instance.w_geq[j+1].value, '*', X.columns[j],'>=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('t_' + str(j+1), '=', instance.t[j+1].value)
            print("\n")
            print('w_0 =', instance.w0.value)
            print("\n")
            
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('b_' + str(i+1) + '_' + str(j+1) + '=', instance.b[j+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_leq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_leq[i+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_geq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_geq[i+1,j+1].value)
            
            print("\n")
            
            
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)
            print('time =', T)
            objective_value_solver = instance.error()
            print('objective_value_solver', objective_value_solver)
            
            ###############
            # ACC OMEGA
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_alpha = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            r_b = np.zeros((n,p))
            r_omega_leq = np.zeros((n,p))
            r_omega_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                r_alpha[j] = instance.alpha[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
                    r_b[i,j] = instance.b[i+1,j+1].value
                    r_omega_leq[i,j] = instance.omega_leq[i+1,j+1].value
                    r_omega_geq[i,j] = instance.omega_geq[i+1,j+1].value
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq[i,j] + r_omega_geq[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(y==finalpred)/n 
            print('acc_omega', acc_omega)
            
            auc_omega = roc_auc_score(y, pred_omega)
            print('auc_omega', auc_omega)
            
            ###############
            
            # ###############
            # ACC WC
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
            pred_wc = np.zeros(n)
            for i in range(n):
                pred_wc[i] = pred_wc[i] + r_w0
                for j in range(p):
                    pred_wc[i] = pred_wc[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                pred_wc[i] = 1/(1+np.exp(-pred_wc[i]))
            
            finalpred = np.round(pred_wc,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
                   
            acc_wc = sum(y==finalpred)/n
            print('acc_wc', acc_wc)
            
            auc_wc = roc_auc_score(y, pred_wc)
            print('auc_wc', auc_wc)
            # ###############
            
            print("\n")
            
            # ACC train
            pred = np.zeros(n)
            for i in range(n):
                pred[i] = pred[i] + w0
                for j in range(p):
            #        pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*is_geq[i,j]
                    pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*(1-is_leq[i,j])
                pred[i] = 1/(1+np.exp(-pred[i]))
            
            finalpred = np.round(pred,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            acc_OptimalSolution = sum(y==finalpred)/n
            print('acc_OptimalSolution', acc_OptimalSolution)
            
            auc_OptimalSolution = roc_auc_score(y, pred)
            print('auc_OptimalSolution', auc_OptimalSolution)
            
            # Objective value
            test = 0
            for i in range(n):
                term_leq = 1*(x[i,:] <= t)
                #term_leq = 1*(X.iloc[i,:] <= t)
                #print(np.sum(w_leq*term_leq))
                term_geq = 1*(x[i,:] >= t)
                #print(sum(term_leq+term_geq))
                #term_geq = 1*(x[i,:] >= t)
                #print(np.sum(w_geq*term_geq))
               # test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*term_leq) + np.sum(w_geq*term_geq) + w0)))
                test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*is_leq[i,:]) + np.sum(w_geq*is_geq[i,:]) + w0)))
            
            print('objective_value_OptimalSolution', test)
            
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0"] = r_w0
            shelf["r_w_leq"] = r_w_leq
            shelf["r_w_geq"] = r_w_geq
            shelf["r_t"] = r_t
            shelf["r_alpha"] = r_alpha
            shelf["r_b"] = r_b
            shelf["r_check_leq"] = r_check_leq
            shelf["r_check_geq"] = r_check_geq
            shelf["r_beta_leq"] = r_omega_leq
            shelf["r_beta_geq"] = r_omega_geq
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["w0"] = w0
            shelf["w_leq"] = w_leq
            shelf["w_geq"] = w_geq
            shelf["t"] = t 
            shelf["X"] = X
            shelf["y"] = y 
            shelf["X0"] = X0 
            shelf["T"] = T 
            shelf["objective_value_solver"] = objective_value_solver
            shelf["pred_omega"] = pred_omega
            shelf["acc_omega"] = acc_omega
            shelf["auc_omega"] = auc_omega
            shelf["pred_wc"] = pred_wc
            shelf["acc_wc"] = acc_wc  
            shelf["auc_wc"] = auc_wc  
            shelf["pred_OptimalSolution)"] = pred
            shelf["acc_OptimalSolution"] = acc_OptimalSolution
            shelf["objective_value_OptimalSolution"] = test
            shelf["auc_OptimalSolution"] = auc_OptimalSolution
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()
        
# Experiment 4
for which_solver in ["bonmin","ipopt"]:
    TL = 1000
    
    for n in [1000,5000,10000,15000,20000]:
        np.random.seed(0)
        #n = 200
        p = 2
        k0 = 2
        
        a = 0
        b = 1
        
        w0 = 0
        
        if p==1:    
            w_leq = np.array([2]) #p1
            w_geq = np.array([0]) #p1
            t = np.array([0.4]) #p1
        elif p ==2:    
            w_leq = np.array([2,0]) #p2
            w_geq = np.array([0,-5]) #p2
            t = np.array([0.4,0.2]) #p2
        elif p ==3:
            w_leq = np.array([2,0,3]) #p3
            w_geq = np.array([0,-5,0]) #p3
            t = np.array([0.4,0.2,0.7]) #p3
        elif p == 4:
            w_leq = np.array([2,0,3,-1]) #p4
            w_geq = np.array([0,-5,0,0]) #p4
            t = np.array([0.4,0.2,0.7,0.8]) #p4
        elif p ==5:
            w_leq = np.array([2,0,3,-1,0]) #p5
            w_geq = np.array([0,-5,0,0,4]) #p5
            t = np.array([0.4,0.2,0.7,0.8,0.1]) #p5
        x = np.zeros((n, p))
        for j in range(p):
            x[:,j] = np.random.uniform(a,b,n)
        
        y = np.ones(n)
        is_leq = np.zeros((n,p))
        is_geq = np.zeros((n,p))
        out = []
        proba = []
        for i in range(n):
            for j in range(p):
                is_leq[i,j] = int(x[i,j] <= t[j])
                is_geq[i,j] = int(x[i,j] >= t[j])
                out.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0))==0.5)
                proba.append(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)))
            if np.round(1/(1+np.exp(-np.sum(w_leq*is_leq[i,:]) - np.sum(w_geq*is_geq[i,:]) - w0)),0) == 0:
                y[i] = -1
        
        xy = np.zeros((n,p+1))
        xy[:,0:p] = x[:,0:p]
        xy[:,p] = y
        
        path = "C:/Users/crist/Desktop/PARIS/RuleSets"  
        os.chdir(path)
        
        df = pd.DataFrame(xy)
        listvars = []
        for j in range(p):
            listvars.append('v'+str(j+1))
        listvars.append('Class')
        df.columns = listvars
        fea = df.columns[0:(len(df.columns)-1)]
        X = df[fea] # Features
        y = df.Class # Target variable
        
        # Normalization to the 0-1 interval
        minimum = {}
        maximum = {}
        for j in range(len(X.columns)):
            minimum[j] = min(X[X.columns[j]])
            maximum[j] = max(X[X.columns[j]])
            for i in range(0,len(X)):
                X.iloc[i,j] = (X.iloc[i,j]-minimum[j])/(maximum[j]-minimum[j])
        
        X0={}
        for i in range(0,len(X)):
            for j in range(len(X.columns)):
                X0[i+1,j+1]=X.iloc[i][j]
                
        y0={}
        for i in range(0,len(y)):
            if y[i]==0:
                y[i] += -1
            y0[i+1]=list(y)[i]
        
        data = {None: dict(
                n = {None : len(X)},
                p = {None : len(X.columns)},
                k = {None : k0},
                X = X0,
                Y = y0)
                }
        
        import time
        
        from model_new import model
        
        if which_solver == "ipopt":
            pathsolver = "C:/coin/ipopt.exe"
            opt = SolverFactory("ipopt",executable=pathsolver,time_limit=TL)
        else:
            pathsolver = "C:/coin/bonmin.exe"
            opt = SolverFactory("bonmin",executable=pathsolver)
            opt.set_options('bonmin.time_limit=' +str(TL) + ' bonmin.algorithm=B-Hyb')
        
        instance = model.create_instance(data) 
        
        # This is code for giving an initial solution:
        #instance.w0 =  w0
        #for i in range(0,len(X)):
        #    for j in range(len(X.columns)):
        #        instance.b[i+1,j+1] = is_leq[i,j]
        #        instance.omega_leq[i+1,j+1] = is_leq[i,j]*w_leq[j]
        #        instance.omega_geq[i+1,j+1] = is_geq[i,j]*w_geq[j]      
        #for j in range(len(X.columns)):
        #    instance.w_leq[j+1] = w_leq[j]
        #    instance.w_geq[j+1] = w_geq[j]
        #    instance.t[j+1] = t[j]
        #    instance.alpha[j+1] = 1
        
        import sys
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f
            
        t0 = time.time()
        results = opt.solve(instance,tee=True,keepfiles=True, logfile = path + "/results_auc/" +  which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + "_log.log")
        t1 = time.time()
        T = t1-t0
        
        try:
            instance.solutions.load_from(results)
            print("\n")
            for j in range(len(X.columns)):
                print('alpha_' + str(j+1), '=', instance.alpha[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_leq_' + str(j+1), '=', instance.w_leq[j+1].value, '*', X.columns[j],'<=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('w_geq_' + str(j+1), '=', instance.w_geq[j+1].value, '*', X.columns[j],'>=',instance.t[j+1].value)
            print("\n")
            for j in range(len(X.columns)):
                print('t_' + str(j+1), '=', instance.t[j+1].value)
            print("\n")
            print('w_0 =', instance.w0.value)
            print("\n")
            
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('b_' + str(i+1) + '_' + str(j+1) + '=', instance.b[j+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_leq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_leq[i+1,j+1].value)
            print("\n")
            for i in range(0,len(X)):
                for j in range(len(X.columns)):
                    print('beta_geq_' + str(i+1) + '_' + str(j+1) + '=', instance.omega_geq[i+1,j+1].value)
            
            print("\n")
            
            
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)
            print('time =', T)
            objective_value_solver = instance.error()
            print('objective_value_solver', objective_value_solver)
            
            ###############
            # ACC OMEGA
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_alpha = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            r_b = np.zeros((n,p))
            r_omega_leq = np.zeros((n,p))
            r_omega_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                r_alpha[j] = instance.alpha[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
                    r_b[i,j] = instance.b[i+1,j+1].value
                    r_omega_leq[i,j] = instance.omega_leq[i+1,j+1].value
                    r_omega_geq[i,j] = instance.omega_geq[i+1,j+1].value
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq[i,j] + r_omega_geq[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(y==finalpred)/n 
            print('acc_omega', acc_omega)
            
            auc_omega = roc_auc_score(y, pred_omega)
            print('auc_omega', auc_omega)
            
            ###############
            
            # ###############
            # ACC WC
            p = len(X.columns)
            n = len(X)
            r_w0 = instance.w0.value
            r_w_leq = np.zeros(p)
            r_w_geq = np.zeros(p)
            r_t = np.zeros(p)
            r_check_leq = np.zeros((n,p))
            r_check_geq = np.zeros((n,p))
            for j in range(p):
                r_w_leq[j] = instance.w_leq[j+1].value
                r_w_geq[j] = instance.w_geq[j+1].value
                r_t[j] = instance.t[j+1].value
                for i in range(n):
                    if X.iloc[i][j] <= r_t[j]:
                        r_check_leq[i,j] = 1
                    if X.iloc[i][j] >= r_t[j]:
                        r_check_geq[i,j] = 1
            pred_wc = np.zeros(n)
            for i in range(n):
                pred_wc[i] = pred_wc[i] + r_w0
                for j in range(p):
                    pred_wc[i] = pred_wc[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                pred_wc[i] = 1/(1+np.exp(-pred_wc[i]))
            
            finalpred = np.round(pred_wc,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
                   
            acc_wc = sum(y==finalpred)/n
            print('acc_wc', acc_wc)
            
            auc_wc = roc_auc_score(y, pred_wc)
            print('auc_wc', auc_wc)
            # ###############
            
            print("\n")
            
            # ACC train
            pred = np.zeros(n)
            for i in range(n):
                pred[i] = pred[i] + w0
                for j in range(p):
            #        pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*is_geq[i,j]
                    pred[i] = pred[i] + w_leq[j]*is_leq[i,j] + w_geq[j]*(1-is_leq[i,j])
                pred[i] = 1/(1+np.exp(-pred[i]))
            
            finalpred = np.round(pred,0)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            acc_OptimalSolution = sum(y==finalpred)/n
            print('acc_OptimalSolution', acc_OptimalSolution)
            
            auc_OptimalSolution = roc_auc_score(y, pred)
            print('auc_OptimalSolution', auc_OptimalSolution)
            
            # Objective value
            test = 0
            for i in range(n):
                term_leq = 1*(x[i,:] <= t)
                #term_leq = 1*(X.iloc[i,:] <= t)
                #print(np.sum(w_leq*term_leq))
                term_geq = 1*(x[i,:] >= t)
                #print(sum(term_leq+term_geq))
                #term_geq = 1*(x[i,:] >= t)
                #print(np.sum(w_geq*term_geq))
                # test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*term_leq) + np.sum(w_geq*term_geq) + w0)))
                test = test + np.log(1 + np.exp(-y[i]*(np.sum(w_leq*is_leq[i,:]) + np.sum(w_geq*is_geq[i,:]) + w0)))
            
            print('objective_value_OptimalSolution', test)
            
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0"] = r_w0
            shelf["r_w_leq"] = r_w_leq
            shelf["r_w_geq"] = r_w_geq
            shelf["r_t"] = r_t
            shelf["r_alpha"] = r_alpha
            shelf["r_b"] = r_b
            shelf["r_check_leq"] = r_check_leq
            shelf["r_check_geq"] = r_check_geq
            shelf["r_beta_leq"] = r_omega_leq
            shelf["r_beta_geq"] = r_omega_geq
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["w0"] = w0
            shelf["w_leq"] = w_leq
            shelf["w_geq"] = w_geq
            shelf["t"] = t 
            shelf["X"] = X
            shelf["y"] = y 
            shelf["X0"] = X0 
            shelf["T"] = T 
            shelf["objective_value_solver"] = objective_value_solver
            shelf["pred_omega"] = pred_omega
            shelf["acc_omega"] = acc_omega
            shelf["auc_omega"] = auc_omega
            shelf["pred_wc"] = pred_wc
            shelf["acc_wc"] = acc_wc  
            shelf["auc_wc"] = auc_wc  
            shelf["pred_OptimalSolution)"] = pred
            shelf["acc_OptimalSolution"] = acc_OptimalSolution
            shelf["objective_value_OptimalSolution"] = test
            shelf["auc_OptimalSolution"] = auc_OptimalSolution
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()