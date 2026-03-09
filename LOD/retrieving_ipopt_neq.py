# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:56:03 2024

@author: crist
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

# Experiment 1
for which_solver in ["ipopt"]:
    TL = 1000
    for p in [1,2,3,4,5]: 
        n = 200
        #p = 1
        k0 = p
        path = "/home/molero/RuleSets"  
        os.chdir(path)
        
        import sys
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f    
        try:                 
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)

            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            r_w0 = shelf["r_w0"]
            r_w_leq = shelf["r_w_leq"] 
            r_w_geq = shelf["r_w_geq"]
            r_t = shelf["r_t"] 
            r_alpha = shelf["r_alpha"] 
            r_X = shelf["X"]
            r_y = shelf["y"]
            shelf.close()      

            r_w0_round = np.round(r_w0,0)
            r_w_leq_round = np.zeros(p)
            r_w_geq_round = np.zeros(p)
            r_t_round = r_t
            r_alpha_round = np.zeros(p)
            r_check_leq_round = np.zeros((n,p))
            r_check_geq_round = np.zeros((n,p))
            r_b_round = np.zeros((n,p))
            r_omega_leq_round = np.zeros((n,p))
            r_omega_geq_round = np.zeros((n,p))

            r_alpha_argsort = np.argsort(r_alpha)[::-1]
            index = 0
            while index < k0:
                r_alpha_round[r_alpha_argsort[index]] = np.ceil(r_alpha[r_alpha_argsort[index]])
                index = index + 1                
            for j in range(p):
                if r_alpha_round[j] == 0:
                    r_w_leq_round[j] = 0
                    r_w_geq_round[j] = 0
                else:
                    if r_w_leq[j]>=0:
                        r_w_leq_round[j] = np.round(r_w_leq[j],0)
                    else:
                        r_w_leq_round[j] = np.floor(r_w_leq[j])
                    if r_w_geq[j]>=0:
                        r_w_geq_round[j] = np.round(r_w_geq[j],0)
                    else:
                        r_w_geq_round[j] = np.floor(r_w_geq[j])
                for i in range(n):
                    if r_X.iloc[i][j] <= r_t_round[j]:
                        r_check_leq_round[i,j] = 1
                    if r_X.iloc[i][j] >= r_t_round[j]:
                        r_check_geq_round[i,j] = 1
                    r_b_round[i,j] = r_check_leq_round[i,j]
                    r_omega_leq_round[i,j] = r_b_round[i,j]*r_w_leq_round[j]
                    r_omega_geq_round[i,j] = (1-r_b_round[i,j])*r_w_geq_round[j]
            ###############
            # ACC
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0_round
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq_round[i,j] + r_omega_geq_round[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(r_y==finalpred)/n 
            print("\n")
            print('acc_round', acc_omega)
            
            auc_omega = roc_auc_score(r_y, pred_omega)
            print('auc_round', auc_omega)
            
            ###############
            
            
                        
            # Objective value
            test = 0
            for i in range(n):
                test = test + np.log(1 + np.exp(-r_y[i]*(np.sum(r_omega_leq_round[i,:]) + np.sum(r_omega_geq_round[i,:]) + r_w0_round)))
            print('objective_value_round', test)
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" +  which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0_round"] = r_w0_round
            shelf["r_w_leq_round"] = r_w_leq_round
            shelf["r_w_geq_round"] = r_w_geq_round
            shelf["r_t_round"] = r_t_round
            shelf["r_alpha_round"] = r_alpha_round
            shelf["r_b_round"] = r_b_round
            shelf["r_check_leq_round"] = r_check_leq_round
            shelf["r_check_geq_round"] = r_check_geq_round
            shelf["r_beta_leq_round"] = r_omega_leq_round
            shelf["r_beta_geq_round"] = r_omega_geq_round
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["pred_omega_wc"] = pred_omega
            shelf["acc_omega_wc"] = acc_omega
            shelf["auc_omega_wc"] = auc_omega 
            shelf["objective_value_round"] = test
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()

# Experiment 2
for which_solver in ["ipopt"]:
    TL = 1000   
    for k0 in [1,2,3,4]:
        n = 200
        p = 5
        #k0 = p
        path = "/home/molero/RuleSets"  
        os.chdir(path)
        
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f    
        try:                 
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)

            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            r_w0 = shelf["r_w0"]
            r_w_leq = shelf["r_w_leq"] 
            r_w_geq = shelf["r_w_geq"]
            r_t = shelf["r_t"] 
            r_alpha = shelf["r_alpha"] 
            r_X = shelf["X"]
            r_y = shelf["y"]
            shelf.close()      

            print("\n")
            print("sum_y", np.sum(r_y)/n)
            print("\n")

            r_w0_round = np.round(r_w0,0)
            r_w_leq_round = np.zeros(p)
            r_w_geq_round = np.zeros(p)
            r_t_round = r_t
            r_alpha_round = np.zeros(p)
            r_check_leq_round = np.zeros((n,p))
            r_check_geq_round = np.zeros((n,p))
            r_b_round = np.zeros((n,p))
            r_omega_leq_round = np.zeros((n,p))
            r_omega_geq_round = np.zeros((n,p))

            r_alpha_argsort = np.argsort(r_alpha)[::-1]
            index = 0
            while index < k0:
                r_alpha_round[r_alpha_argsort[index]] = np.ceil(r_alpha[r_alpha_argsort[index]])
                index = index + 1                
            for j in range(p):
                if r_alpha_round[j] == 0:
                    r_w_leq_round[j] = 0
                    r_w_geq_round[j] = 0
                else:
                    if r_w_leq[j]>=0:
                        r_w_leq_round[j] = np.round(r_w_leq[j],0)
                    else:
                        r_w_leq_round[j] = np.floor(r_w_leq[j])
                    if r_w_geq[j]>=0:
                        r_w_geq_round[j] = np.round(r_w_geq[j],0)
                    else:
                        r_w_geq_round[j] = np.floor(r_w_geq[j])
                for i in range(n):
                    if r_X.iloc[i][j] <= r_t_round[j]:
                        r_check_leq_round[i,j] = 1
                    if r_X.iloc[i][j] >= r_t_round[j]:
                        r_check_geq_round[i,j] = 1
                    r_b_round[i,j] = r_check_leq_round[i,j]
                    r_omega_leq_round[i,j] = r_b_round[i,j]*r_w_leq_round[j]
                    r_omega_geq_round[i,j] = (1-r_b_round[i,j])*r_w_geq_round[j]
            ###############
            # ACC
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0_round
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq_round[i,j] + r_omega_geq_round[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(r_y==finalpred)/n 
            print("\n")
            print('acc_round', acc_omega)
            
            auc_omega = roc_auc_score(r_y, pred_omega)
            print('auc_round', auc_omega)
            
            ###############
            
            
                        
            # Objective value
            test = 0
            for i in range(n):
                test = test + np.log(1 + np.exp(-r_y[i]*(np.sum(r_omega_leq_round[i,:]) + np.sum(r_omega_geq_round[i,:]) + r_w0_round)))
            print('objective_value_round', test)
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" +  which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0_round"] = r_w0_round
            shelf["r_w_leq_round"] = r_w_leq_round
            shelf["r_w_geq_round"] = r_w_geq_round
            shelf["r_t_round"] = r_t_round
            shelf["r_alpha_round"] = r_alpha_round
            shelf["r_b_round"] = r_b_round
            shelf["r_check_leq_round"] = r_check_leq_round
            shelf["r_check_geq_round"] = r_check_geq_round
            shelf["r_beta_leq_round"] = r_omega_leq_round
            shelf["r_beta_geq_round"] = r_omega_geq_round
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["pred_omega_wc"] = pred_omega
            shelf["acc_omega_wc"] = acc_omega
            shelf["auc_omega_wc"] = auc_omega 
            shelf["objective_value_round"] = test
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()
        
# Experiment 3
for which_solver in ["ipopt"]:
    TL = 1000    
    # for n in [1000,2000,3000,4000,5000,10000,11000,12000,13000,14000,15000,20000]:
    for n in [1000,5000,10000,15000,20000]:
        #n = 200
        p = 1
        k0 = 1
        path = "/home/molero/RuleSets"  
        os.chdir(path)
        
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f    
        try:                 
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)

            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            r_w0 = shelf["r_w0"]
            r_w_leq = shelf["r_w_leq"] 
            r_w_geq = shelf["r_w_geq"]
            r_t = shelf["r_t"] 
            r_alpha = shelf["r_alpha"] 
            r_X = shelf["X"]
            r_y = shelf["y"]
            shelf.close()      

            r_w0_round = np.round(r_w0,0)
            r_w_leq_round = np.zeros(p)
            r_w_geq_round = np.zeros(p)
            r_t_round = r_t
            r_alpha_round = np.zeros(p)
            r_check_leq_round = np.zeros((n,p))
            r_check_geq_round = np.zeros((n,p))
            r_b_round = np.zeros((n,p))
            r_omega_leq_round = np.zeros((n,p))
            r_omega_geq_round = np.zeros((n,p))

            r_alpha_argsort = np.argsort(r_alpha)[::-1]
            index = 0
            while index < k0:
                r_alpha_round[r_alpha_argsort[index]] = np.ceil(r_alpha[r_alpha_argsort[index]])
                index = index + 1                
            for j in range(p):
                if r_alpha_round[j] == 0:
                    r_w_leq_round[j] = 0
                    r_w_geq_round[j] = 0
                else:
                    if r_w_leq[j]>=0:
                        r_w_leq_round[j] = np.round(r_w_leq[j],0)
                    else:
                        r_w_leq_round[j] = np.floor(r_w_leq[j])
                    if r_w_geq[j]>=0:
                        r_w_geq_round[j] = np.round(r_w_geq[j],0)
                    else:
                        r_w_geq_round[j] = np.floor(r_w_geq[j])
                for i in range(n):
                    if r_X.iloc[i][j] <= r_t_round[j]:
                        r_check_leq_round[i,j] = 1
                    if r_X.iloc[i][j] >= r_t_round[j]:
                        r_check_geq_round[i,j] = 1
                    r_b_round[i,j] = r_check_leq_round[i,j]
                    r_omega_leq_round[i,j] = r_b_round[i,j]*r_w_leq_round[j]
                    r_omega_geq_round[i,j] = (1-r_b_round[i,j])*r_w_geq_round[j]
            ###############
            # ACC
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0_round
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq_round[i,j] + r_omega_geq_round[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(r_y==finalpred)/n 
            print("\n")
            print('acc_round', acc_omega)
            
            auc_omega = roc_auc_score(r_y, pred_omega)
            print('auc_round', auc_omega)
            
            ###############
            
            
                        
            # Objective value
            test = 0
            for i in range(n):
                test = test + np.log(1 + np.exp(-r_y[i]*(np.sum(r_omega_leq_round[i,:]) + np.sum(r_omega_geq_round[i,:]) + r_w0_round)))
            print('objective_value_round', test)
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" +  which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0_round"] = r_w0_round
            shelf["r_w_leq_round"] = r_w_leq_round
            shelf["r_w_geq_round"] = r_w_geq_round
            shelf["r_t_round"] = r_t_round
            shelf["r_alpha_round"] = r_alpha_round
            shelf["r_b_round"] = r_b_round
            shelf["r_check_leq_round"] = r_check_leq_round
            shelf["r_check_geq_round"] = r_check_geq_round
            shelf["r_beta_leq_round"] = r_omega_leq_round
            shelf["r_beta_geq_round"] = r_omega_geq_round
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["pred_omega_wc"] = pred_omega
            shelf["acc_omega_wc"] = acc_omega
            shelf["auc_omega_wc"] = auc_omega 
            shelf["objective_value_round"] = test
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()
        
# Experiment 4
for which_solver in ["ipopt"]:
    TL = 1000
    #for n in [1000]: 
    for n in [1000,5000,10000,15000,20000]:
        #n = 200
        p = 2
        k0 = 2
        path = "/home/molero/RuleSets"  
        os.chdir(path)
        
        orig_stdout = sys.stdout
        f = open(path + "/results_auc/" + which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p" + str(p) + "_k" + str(k0) + '_all.txt', 'w')
        
        sys.stdout = f    
        try:                 
            print("n =",n)
            print("p =",p)
            print("k =",k0)
            print("\n")
            print("solver =", which_solver)
            print("TL =", TL)

            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" + which_solver + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            r_w0 = shelf["r_w0"]
            r_w_leq = shelf["r_w_leq"] 
            r_w_geq = shelf["r_w_geq"]
            r_t = shelf["r_t"] 
            r_alpha = shelf["r_alpha"] 
            r_X = shelf["X"]
            r_y = shelf["y"]
            shelf.close()      

            r_w0_round = np.round(r_w0,0)
            r_w_leq_round = np.zeros(p)
            r_w_geq_round = np.zeros(p)
            r_t_round = r_t
            r_alpha_round = np.zeros(p)
            r_check_leq_round = np.zeros((n,p))
            r_check_geq_round = np.zeros((n,p))
            r_b_round = np.zeros((n,p))
            r_omega_leq_round = np.zeros((n,p))
            r_omega_geq_round = np.zeros((n,p))

            r_alpha_argsort = np.argsort(r_alpha)[::-1]
            index = 0
            while index < k0:
                r_alpha_round[r_alpha_argsort[index]] = np.ceil(r_alpha[r_alpha_argsort[index]])
                index = index + 1                
            for j in range(p):
                if r_alpha_round[j] == 0:
                    r_w_leq_round[j] = 0
                    r_w_geq_round[j] = 0
                else:
                    if r_w_leq[j]>=0:
                        r_w_leq_round[j] = np.round(r_w_leq[j],0)
                    else:
                        r_w_leq_round[j] = np.floor(r_w_leq[j])
                    if r_w_geq[j]>=0:
                        r_w_geq_round[j] = np.round(r_w_geq[j],0)
                    else:
                        r_w_geq_round[j] = np.floor(r_w_geq[j])
                for i in range(n):
                    if r_X.iloc[i][j] <= r_t_round[j]:
                        r_check_leq_round[i,j] = 1
                    if r_X.iloc[i][j] >= r_t_round[j]:
                        r_check_geq_round[i,j] = 1
                    r_b_round[i,j] = r_check_leq_round[i,j]
                    r_omega_leq_round[i,j] = r_b_round[i,j]*r_w_leq_round[j]
                    r_omega_geq_round[i,j] = (1-r_b_round[i,j])*r_w_geq_round[j]
            ###############
            # ACC
            pred_omega = np.zeros(n)
            for i in range(n):
                pred_omega[i] = pred_omega[i] + r_w0_round
                for j in range(p):
            #        pred[i] = pred[i] + r_w_leq[j]*r_check_leq[i,j] + r_w_geq[j]*r_check_geq[i,j]
                    pred_omega[i] = pred_omega[i] + r_omega_leq_round[i,j] + r_omega_geq_round[i,j]
                pred_omega[i] = 1/(1+np.exp(-pred_omega[i]))
            
            finalpred = np.round(pred_omega,0)
            #finalpred = np.floor(pred)
            for i in range(n):
                if finalpred[i] == 0:
                    finalpred[i] = finalpred[i] -1
            
            acc_omega = sum(r_y==finalpred)/n 
            print("\n")
            print('acc_round', acc_omega)
            
            auc_omega = roc_auc_score(r_y, pred_omega)
            print('auc_round', auc_omega)
            
            ###############
            
            
                        
            # Objective value
            test = 0
            for i in range(n):
                test = test + np.log(1 + np.exp(-r_y[i]*(np.sum(r_omega_leq_round[i,:]) + np.sum(r_omega_geq_round[i,:]) + r_w0_round)))
            print('objective_value_round', test)
            
            import shelve
            shelf = shelve.open(path + "/results_auc/" + "results_" +  which_solver + "_round" + "_TL" + str(TL) + "_n" + str(n) + "_p"  + str(p) + "_k" + str(k0) + ".dat")
            shelf["r_w0_round"] = r_w0_round
            shelf["r_w_leq_round"] = r_w_leq_round
            shelf["r_w_geq_round"] = r_w_geq_round
            shelf["r_t_round"] = r_t_round
            shelf["r_alpha_round"] = r_alpha_round
            shelf["r_b_round"] = r_b_round
            shelf["r_check_leq_round"] = r_check_leq_round
            shelf["r_check_geq_round"] = r_check_geq_round
            shelf["r_beta_leq_round"] = r_omega_leq_round
            shelf["r_beta_geq_round"] = r_omega_geq_round
            shelf["n"] = n
            shelf["p"] = p
            shelf["k"] = k0
            shelf["which_solver"] = which_solver 
            shelf["pred_omega_wc"] = pred_omega
            shelf["acc_omega_wc"] = acc_omega
            shelf["auc_omega_wc"] = auc_omega 
            shelf["objective_value_round"] = test
            shelf.close()       
        except:
            print("\n")
            print("EXCEPTION")
        sys.stdout = orig_stdout
        f.close()

print("It worked!")