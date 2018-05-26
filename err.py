import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def error_count(predicted, real, toler_treshold = 5.0):
    
    pred_sz = predicted.shape[0]
    real_sz = real.shape[0]
    
    assert pred_sz == real_sz
    
    # Wrong predicted count
    err_cnt = 0
    
    for i in range(0, real_sz):
        if abs(predicted[i] - real[i]) <= (toler_treshold/100) * real[i] :
            pass
        else:
            err_cnt +=1
            
    return err_cnt


def calc_diff(predicted, real):
    
    pred_sz = predicted.shape[0]
    real_sz = real.shape[0]
    
    assert pred_sz == real_sz
    
    # Calc difference between real data price and predicted price
    diff_rate = np.zeros(real_sz)

    for i in range(0, real_sz):
        diff_rate[i] = abs(predicted[i] - real[i])
        
    return diff_rate


def mean_absolute_percentage_error(y_true, y_pred): 
    
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

