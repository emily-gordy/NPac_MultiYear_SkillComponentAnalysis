import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import eig

class SCA:

    def __init__(self,y_pred_val,outputval,landmask,weights):
        
        self.landmask = landmask
            
        y_pred_val_func = y_pred_val*weights[np.newaxis,:,:]
        outputval_func = outputval*weights[np.newaxis,:,:]

        predrectangle = y_pred_val_func[:,self.landmask]
        verrectangle = outputval_func[:,self.landmask]
        errrectangle = predrectangle-verrectangle
        
        Se = np.cov(np.transpose(errrectangle))
        Sv = np.cov(np.transpose(verrectangle))
        
        eigvals,evecs = eig(Se,Sv)
        bestinds = np.argsort(np.real(eigvals))

        ivec = 0
        
        evecsel = evecs[:,bestinds[ivec]]
        
        if np.nansum(evecsel)<0:
            evecsel = -1*evecsel
        truecomponent = (1/len(evecsel))*np.matmul(evecsel,np.transpose(verrectangle))
        #standardize component
        truecomponent = (truecomponent-np.mean(truecomponent))/np.std(truecomponent)
        
        bestpattern = (1/len(truecomponent))*np.matmul(np.transpose(verrectangle),truecomponent)
        bestpattern_out = np.empty((outputval.shape[1],outputval.shape[2]))+np.nan
        bestpattern_out[self.landmask]=bestpattern

        self.bestpattern = bestpattern_out

    def index_timeseries(self,data):
    
        data_rect = data[:,self.landmask]
        patternvec = self.bestpattern[self.landmask]
        index = np.matmul(data_rect,patternvec)
        
        index = (index-np.mean(index))/np.std(index)

        return index
    
    def corr_indextimeseries(self,truedata,preddata):

        SC_index_true = self.index_timeseries(truedata,self.landmask)
        SC_index_pred = self.index_timeseries(preddata,self.landmask)
        
        r,p = pearsonr(SC_index_true,SC_index_pred)
        
        return r,p
    