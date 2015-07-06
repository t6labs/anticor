# -*- coding: utf-8 -*-
"""
@author: T6Labs

Anticor is an aggressive mean reversion portfolio selection algorithm created by A. Borodin, R. El-Yaniv, and V. Gogan.  A vectorized implementation was used where ever possible. See https://www.t6labs.com/a_winning_portfolio_selection_algorithm for more details

"""

import numpy as np

class Algo:
    def anticor(self,w,t,x,b,lt):
#Inputs: w = window size, t = index of last trading day, x = Historical market sequence of returns (p / p_[t-1]), b_hat = current portfolio (at end of trading day t).  Initialize m to width of matrix x (= number of stocks)

        if t < (2 * w) - 1:
            return b
        else:
             
#Retrive data from appropriate windows and convert returns data to logs.  Interestingly, the algorithm sometimes works better without the log transformation.  The lt switch passed to the function determines whether we use this or not.
            
            if lt:
                LX1 = np.log(x[t-(2*w) + 1:t - w + 1,:])
                LX2 = np.log(x[t-w + 1:t + 1,:])
            else:    
                LX1 = x[t-(2*w) + 1:t - w + 1,:]
                LX2 = x[t-w + 1:t + 1,:]
            
#Calculate mean and standard deviation of each column, mu1, sig1 is mean, standard deviation of each stock from window 1, mu2, sig2 is mean, standard deviation of each stock from window 2, sigma is dot product of sig1 and sig2 used for correlation calculation below (mxm)
            
            mu1 = np.average(LX1, axis=0)
            sig1 = np.std(LX1, axis=0, ddof=1)
            mu2 = np.average(LX2, axis=0)
            sig2 = np.std(LX2, axis=0, ddof=1)
            sigma = np.outer(np.transpose(sig1),sig2)
            
# Create boolean matrix to compare mu2[i] to mu2[j]

            mu_matrix = np.ones((mu2.shape[0],mu2.shape[0]), dtype = bool)
            for i in range(0, mu2.shape[0]):
                for j in range(0, mu2.shape[0]):
                    if mu2[i] > mu2[j]:
                        mu_matrix[i,j] = True
                    else:
                        mu_matrix[i,j] = False           
            
#Covariance matrix is dot product of x - mu of window 1 and window 2 (mxm)
            
            mCov = (1.0/np.float64(w-1)) * np.dot(np.transpose(np.subtract(LX1,mu1)),np.subtract(LX2,mu2)) 

#Correlation matrix is mCov divided element wise by sigma (mxm), 0 if sig1, sig2 = 0

            mCorr = np.where(sigma != 0, np.divide(mCov,sigma), 0)
            
#Multiply the correlation matrix by the boolean matrix comparing mu2[i] to mu2[j] and by the boolean matrix where correlation matrix is greater than zero
            
            claim = np.multiply(mCorr,np.multiply(mCorr > 0,mu_matrix))             

#The boolean claim matrix will be used to obtain only the entries that meet the criteria that mu2[i] > mu2[j] and mCorr is > 0 for the i_corr and j_corr matrices
            
            bool_claim = claim > 0
            
#If stock i is negatively correlated with itself we want to add that correlation to all instances of i.  To do this, we multiply a matrix of ones by the diagonal of the correlation matrix row wise.
            
            i_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[:,np.newaxis])  
            
#Since our condition is when the correlation is negative, we zero out any positive values, also we want to multiply by the bool_claim matrix to obtain valid entries only
            
            i_corr = np.where(i_corr > 0,0,i_corr)
            i_corr = np.multiply(i_corr,bool_claim)
            
#Subtracting out these negative correlations is essentially the same as adding them to the claims matrix
            
            claim -= i_corr
            
#We repeat the same process for stock j except this time we will multiply the diagonal of the correlation matrix column wise
            
            j_corr = np.multiply(np.ones((mu1.shape[0],mu2.shape[0])),np.diagonal(mCorr)[np.newaxis,:]) 

#Since our condition is when the correlation is negative, we zero out any positive values again multiplying by the bool_claim matrix
           
            j_corr = np.where(j_corr > 0,0,j_corr)
            j_corr = np.multiply(j_corr,bool_claim)            

#Once again subtract these to obtain our final claims matrix            

            claim -= j_corr   
                                
#Create the wealth transfer matrix first by summing the claims matrix along the rows
                  
            sum_claim = np.sum(claim, axis=1)
            
#Then divide each element of the claims matrix by the sum of it's corresponding row
            
            transfer = np.divide(claim,sum_claim[:,np.newaxis])
            
#Multiply the original weights to get the transfer matrix row wise

            transfer = np.multiply(transfer,b[:,np.newaxis])
            
#Replace the nan with zeros in case the divide encountered any
            
            transfer = np.where(np.isnan(transfer),0,transfer)                        
 
#We don't transfer any stock to itself, so we zero out the diagonals     
      
            np.fill_diagonal(transfer,0)
                        
#Create the new portfolio weight adjustments, by adding the j direction weights or the sum by columns and subtracting the i direction weights or the sum by rows
                        
            adjustment = np.subtract(np.sum(transfer, axis=0),np.sum(transfer,axis=1))
            
#Finally, we adjust our original weights and we are done
            
            b += adjustment
 
            return b    
