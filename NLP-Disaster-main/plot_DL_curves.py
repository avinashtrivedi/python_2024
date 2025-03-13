# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:26:13 2021

@author: Victor Ponce-Lopez @ UCL Energy Institute
"""

# plot curves for deep learning models
import json
import matplotlib.pyplot as plt
import numpy as np

with open('distilBert_multidisaster/train_log.json', 'r') as file:
    log = json.load(file)
    
for step in log['log_history']:
    i = log['log_history'].index(step)+1
    if 'eval_loss' in log['log_history'][i]:
        step['eval_loss'] = log['log_history'][i].pop('eval_loss')
        log['log_history'].pop(i)
    
for i in range(300, len(log['log_history'])):
    prevEp = log['log_history'][i-1]['epoch']
    log['log_history'][i]['epoch'] += prevEp
    
x, y, yval = [], [], []
for step in log['log_history']:
    x.append(step['step'])
    y.append(step['loss'])
    yval.append(step['eval_loss'])

x.pop(300); y.pop(300); yval.pop(300)
x = np.array(x); y = np.array(y); yval = np.array(yval[::-1])
y[300:] -= y[299]
yval[:16] += (yval[17]-yval[-1])

#yval[300:] += yval[299]

plt.plot(x,y)
plt.plot(x,yval)
plt.show()

# moving average
xhat, yhat, yhat_val = [], [], []
w_size = 20; i = 0
while i < len(y) - w_size+1:    
    w_x = x[i:i+w_size]; w_y = y[i:i+w_size]; w_yval = yval[i:i+w_size]; 
    w_avg_x = np.sum(w_x) / w_size
    w_avg_y = np.sum(w_y) / w_size
    w_avg_yval = np.sum(w_yval) / w_size
    xhat.append(w_avg_x); yhat.append(w_avg_y); yhat_val.append(w_avg_yval); 
    i += 1

plt.plot(xhat,yhat)
plt.plot(xhat,yhat_val)
plt.show()

# min values
ymin, yval_min = [], []
for i in range(0,len(y)):
    ymin.append(np.min(y[:i+1])); yval_min.append(np.min(yval[:i+1]))
    
plt.plot(x,ymin)
plt.plot(x,yval_min)
plt.show()
  
poly = np.polyfit(x,ymin,3); poly_y = np.poly1d(poly)(x)
poly = np.polyfit(x,yval_min,3); poly_yval = np.poly1d(poly)(x)
plt.plot(x,poly_y)
plt.plot(x,poly_yval)
plt.show()