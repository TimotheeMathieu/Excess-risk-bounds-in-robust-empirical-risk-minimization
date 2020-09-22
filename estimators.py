import numpy as np
from sklearn.base import BaseEstimator,clone
import inspect

   
class _huber():
    def __init__(self, beta=None, maxiter=100, tol=1e-6):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def psi(self, x, beta):
        if beta != 0:
            return x * (abs(x) < beta) + (abs(x) >= beta) * (2 * (x > 0) - 1) * beta
        else:
            return 2 * (x > 0) - 1

    def psisx(self, x, beta):
        return (abs(x) < beta) + (abs(x) >= beta) * beta / (1e-10 + np.abs(x))

    def estimate(self, X):
        beta = self.beta
        if beta == 0:
            return np.median(X)
        # Initialization
        mu = np.median(X)
        last_mus = []

        # Iterative Reweighting algorithm
        for f in range(self.maxiter):

            w = self.psisx(np.abs(X - mu), beta)

            mu = np.average(X, axis=0, weights=w/np.sum(w))
            
            # Stopping criterion
            if f>10:
                if np.std(last_mus) < self.tol:
                    break
                else:
                    last_mus.pop(0)
                    last_mus.append(mu)
            else:
                last_mus.append(mu)
                

        self.beta_ = beta
        self.weights_ = w / np.sum(w)

        return mu



def huber(x,c):
    '''Compute Huber estimator on x with tuning parameter c
    '''
    if np.sum(1-np.isfinite(x))>1:
        return np.nan
    else:
        hub = _huber(c)
        return hub.estimate(x)

def blockMOM(K,x):
    '''Sample the indices of K blocks for data x using a random permutation

    Parameters
    ----------

    K : int
        number of blocks

    x : array like, length = n_sample
        sample whose size correspong to the size of the sample we want to do blocks for.

    Returns 
    -------

    list of size K containing the lists of the indices of the blocks, the size of the lists are contained in [n_sample/K,2n_sample/K]
    '''
    b=int(np.floor(len(x)/K))
    nb=K-(len(x)-b*K)
    nbpu=len(x)-b*K
    perm=np.random.permutation(len(x))
    blocks=[[(b+1)*g+f for f in range(b+1) ] for g in range(nbpu)]
    blocks+=[[nbpu*(b+1)+b*g+f for f in range(b)] for g in range(nb)]
    return [perm[b] for  b in blocks]

class classifier(BaseEstimator):
    def __init__( self,w0=None,K=3,eta0=1e-1,beta=1e-3,momentum=0.9,epochs=200,tol=1e-3,step='inverse',Delta=None,shuffle=True):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.c=Delta # Rename to match previous version of algo.

    def psic(self,x,c):
        result=[ xx if np.abs(xx)<=c else self.c*(2*(xx>0)-1) for xx in x]
        return np.array(result)

    def dpsic(self,x,c):
        result=[ 1 if np.abs(xx)<=c else 0 for xx in x]
        return result
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def dmu(self,w,x,y,blocks):
        pertes=np.log(1+np.exp(-x.dot(w)*y))

        pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
        if self.c is not None:
            c=self.c
        else:
            c=max(np.median(np.abs(pertes_b-np.median(pertes_b))),self.tol)
        mut=huber(pertes_b,c)
        psip=self.dpsic(pertes_b-mut,c)
        if np.sum(psip)==0 or ~np.isfinite(mut):
            return np.zeros(len(w)),np.median(pertes_b)
        else:
            return np.array([np.sum([np.mean([x[i][j]*y[i]*self.sigmoid(-x[i].dot(w)*y[i]) for i in blocks[k]]) for k in range(self.K) if psip[k]==1 ])/np.sum(psip) for j in range(len(x[0]))])+self.beta*w,mut

    def fit(self,x,Y):
        X=np.hstack([np.array(x),np.ones(len(x)).reshape(len(x),1)])
        y=Y.copy()
        self.yvals=list(set(y))
        self.classes_=self.yvals
        self.n_classes_=2
        y[y==self.yvals[0]]=-1
        y[y==self.yvals[1]]=1
        if self.w0 is None :
            self.w0=np.zeros(len(X[0]))
        w=self.w0
        pas = lambda t : self.eta0/(1+self.eta0*self.beta*t)
        blocks=blockMOM(self.K,X)
        v=w.copy()
        self.losses=[]
        

        for t in range(self.epochs):
            if self.shuffle:
                blocks=blockMOM(self.K,X)
            grad,l=self.dmu(w,X,y,blocks)
            self.losses+=[l]
            v=self.momentum*v+(1-self.momentum)*pas(t)*grad
            w=w+v
        self.w=w
        self.coef_=w[:-1]

    def predict_proba(self,xtest):
        X=np.hstack([np.array(xtest),np.ones(len(xtest)).reshape(len(xtest),1)])
        pred=np.array([self.sigmoid(X[i].dot(self.w)) for i in range(len(X))])
        #return np.array([[1-p,p] for p in pred])
        return pred
    def predict(self,xtest):
        pred=(self.predict_proba(xtest)>1/2).astype(np.int32)
        return np.array([self.yvals[p] for p in pred])
    def score(self,x,y):
        return  np.mean(self.predict(x)==y)
    
class regressor(BaseEstimator):
    def __init__( self,w0=None,K=3,eta0=1e-1,beta=1e-3,momentum=0.9,epochs=300,tol=1e-3,Delta=None,shuffle=True):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.c=Delta 

    def psic(self,x,c):
        result=[ xx if np.abs(xx)<=c else self.c*(2*(xx>0)-1) for xx in x]
        return np.array(result)

    def dpsic(self,x,c):
        result=[ 1 if np.abs(xx)<=c else 0 for xx in x]
        return result
    
    def dmu(self,w,x,y,blocks):
        pertes=(w.dot(x.T)-y)**2
        pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
        if self.c is not None:
            c=self.c
        else:
            c=max(np.median(np.abs(pertes_b-np.median(pertes_b))),self.tol)
        mut=huber(pertes_b,c)
        psip=self.dpsic(pertes_b-mut,c)
        if np.sum(psip)==0:
            return np.zeros(len(w)),mut
        else:

            return np.array([np.sum([np.mean([ -2*x[i][j]*(np.sum(w*x[i])-y[i]) for i in blocks[k]]) for k in range(self.K) if psip[k]==1])/np.sum(psip) for j in range(len(x[0]))])+self.beta*w,mut

    def fit(self,x,Y):
        X=np.hstack([np.array(x),np.ones(len(x)).reshape(len(x),1)])
        y=Y.copy()
        if self.w0 is None :
            self.w0=np.zeros(len(X[0]))
        w=self.w0
        pas = lambda t : self.eta0/(1+self.beta*self.eta0*t)
        blocks=blockMOM(self.K,X)
        v=w.copy()
        self.losses=[]
        

        for t in range(self.epochs):
            if self.shuffle:
                blocks=blockMOM(self.K,X)
            grad,l=self.dmu(w,X,y,blocks)
            self.losses+=[l]
            v=self.momentum*v+(1-self.momentum)*pas(t)*grad
            w=w+v
        self.w=w
        self.coef_=w[:-1]

    def predict(self,xtest):

        X=np.hstack([np.array(xtest),np.ones(len(xtest)).reshape(len(xtest),1)])
        pred=self.w.dot(X.T)
        return pred
    def score(self,x,y):
        return  np.mean((self.predict(x)-y)**2)
    
    
