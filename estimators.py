import numpy as np
from sklearn.base import BaseEstimator,clone
import inspect

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
    """
    Classifier class for robust classification.
    Usage is similar to scikit-learn classifiers, use fit then predict/predict_proba.
    Parameters
    ----------
    beta : float or None, default = 1
        Parameter of scale in the estimator. If None, use Lepski's method,
        it can be computationally intensive depending on the value of grid.
        Lepski's method in multi-D is not implemented yet.

    w0 : array-like or None, default = None
        Initialization of the coefficients for the estimator. If None, w0 is set to an all-zero vector.
        
    K : int, default = 3
        Number of blocks.
        
    eta0 : float, default = 1e-1
        step_size parameter. The step size is adaptive of the form self.eta0/(1+self.beta*self.eta0*t) as 
        advised by Leon Botton for SGD (according to scikit-learn).
        
    beta : float, default = 1e-3
        L2 regularization parameter.
    
    momentum : float in [0,1], default = 0.9
        momentum parameter for gradient step.
        
    epochs : int, default = 200
        Number of epochs (one epoch = one gradient step).
        
    T : int, default = 20
        number of steps used to estimate Lhat (using different random permutation)
        
    D : 3
        Parameter for rejection sampling. Increase D if there is too much rejection.
        
    c : float, default = 1
        Delta parameter in the article
        

    Returns
    -------
    Classifier class object

    """
    def __init__( self,w0=None,K=3,eta0=1e-1,beta=1e-3,momentum=0.9,epochs=200,T=20,c=1):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)
        self.norm_weight=0

    def psic(self,x,c):
        result=[ xx if np.abs(xx)<=c else self.c*(2*(xx>0)-1) for xx in x]
        return np.array(result)

    def dpsic(self,x,c):
        result=[ 1 if np.abs(xx)<=c else 0 for xx in x]
        return result
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
  
    def dmu(self,w,x,y,t):
        pertes=np.log(1+np.exp(-x.dot(w)*y))
        c=self.c
        reject=True
        compteur=0
        while reject:
            blocks=blockMOM(self.K,pertes)
            pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
            mut=self.Lhat(pertes,blocks)
            weight,numerator=self.weight(pertes_b,mut,t)
            #sample of the uniform variable
            xi=np.random.uniform()*self.D
            reject = xi>weight
            compteur+=1
            if compteur>100:
                print('Attention, seems too much reject, change D')
                break
        self.norm_weight+=numerator
        psip=self.dpsic(pertes_b-mut,c)
        if (np.sum(psip)==0) or (compteur >100) :
            return np.zeros(len(w)),mut
        else:
            return np.array([np.sum([np.mean([x[i][j]*y[i]*self.sigmoid(-x[i].dot(w)*y[i]) for i in blocks[k]]) for k in range(self.K) if psip[k]==1 ])/np.sum(psip) for j in range(len(x[0]))])+self.beta*w,mut

    def Lhat(self,pertes,blocks):
        c=self.c
        pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
        mut=huber(pertes_b,c)
        pas = lambda t: 1/(1+t)
        for f in range(self.T):
            blocks=blockMOM(self.K,pertes)
            pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
            mut=mut-pas(f)*np.mean(self.psic(pertes_b-mut,c))
        return mut
    
    def num_weight(self,pertes_b,mut):
        return np.sum(self.dpsic(pertes_b-mut,self.c))


    def weight(self,pertes_b,mut,t):
        numerator=self.num_weight(pertes_b,mut)
        normalization=(self.norm_weight+numerator)/t
        return numerator/normalization,numerator
            
     
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
        pas = lambda t : self.eta0/(1+self.beta*self.eta0*t)
        blocks=blockMOM(self.K,X)
        v=w.copy()
        self.losses=[]
        
        # Main gradient loop.
        for t in range(self.epochs):
            grad,l=self.dmu(w,X,y,t+1)
            self.losses+=[l]
            v=self.momentum*v+(1-self.momentum)*pas(t)*grad
            w=w+v
        self.w=w
        self.coef_=w[:-1]

    def predict_proba(self,xtest):
        X=np.hstack([np.array(xtest),np.ones(len(xtest)).reshape(len(xtest),1)])
        pred=np.array([self.sigmoid(X[i].dot(self.w)) for i in range(len(X))])
        return pred
    def predict(self,xtest):
        pred=(self.predict_proba(xtest)>1/2).astype(np.int32)
        return np.array([self.yvals[p] for p in pred])
    def score(self,x,y):
        return  np.mean(self.predict(x)==y)
    
class regressor(BaseEstimator):
    """Regressor Class. Regression equivalent of classifier class, see classifier for more information.
    """
    def __init__( self,w0=None,K=3,eta0=1e-1,beta=1e-3,momentum=0.9,epochs=100,T=10,c=1):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

        self.norm_weight=0


    def psic(self,x,c):
        result=[ xx if np.abs(xx)<=c else self.c*(2*(xx>0)-1) for xx in x]
        return np.array(result)

    def dpsic(self,x,c):
        result=[ 1 if np.abs(xx)<=c else 0 for xx in x]
        return result
    
    def dmu(self,w,x,y,t):
        pertes=(w.dot(x.T)-y)**2
        c=self.c

        reject=True
        compteur=0
        while reject:
            blocks=blockMOM(self.K,pertes)
            pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
            mut=self.Lhat(pertes,blocks)
            weight,numerator=self.weight(pertes_b,mut,t)
            #sample of the uniform variable
            xi=np.random.uniform()*self.D
            reject = xi>weight
            compteur+=1
            if compteur>100:
                print('Attention, seems too much reject, change D')
                break
        self.norm_weight+=numerator
        psip=self.dpsic(pertes_b-mut,c)
        if (np.sum(psip)==0) or (compteur >100) :

            return np.zeros(len(w)),mut
        else:
            return np.array([np.sum([np.mean([ -2*x[i][j]*(np.sum(w*x[i])-y[i]) for i in blocks[k]]) for k in range(self.K) if psip[k]==1])/np.sum(psip) for j in range(len(x[0]))])+self.beta*w,mut


    def Lhat(self,pertes,blocks):
        c=self.c
        pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
        est = huber(c)
        mut=est.estimate(pertes_b)
        pas = lambda t: 1/(1+t)
        for f in range(self.T):
            blocks=blockMOM(self.K,pertes)
            pertes_b=np.array([np.mean(pertes[blocks[k]]) for k in range(self.K)])
            mut=mut-pas(f)*np.mean(self.psic(pertes_b-mut,c))
        return mut
    
    def num_weight(self,pertes_b,mut):
        return np.sum(self.dpsic(pertes_b-mut,self.c))


    def weight(self,pertes_b,mut,t):
        numerator=self.num_weight(pertes_b,mut)
        normalization=(self.norm_weight+numerator)/t
        return numerator/normalization,numerator

        
    def fit(self,x,Y):
        X=np.hstack([np.array(x),np.ones(len(x)).reshape(len(x),1)])
        y=Y.copy()
        if self.w0 is None :
            self.w0=np.zeros(len(X[0]))
        w=self.w0
        pas = lambda t : self.eta0/(1+self.beta*self.eta0*t)
        v=w.copy()
        self.losses=[]
        
        # Main gradient loop.
        for t in range(self.epochs):
            grad,l=self.dmu(w,X,y,t+1)
            self.losses+=[l]
            v=self.momentum*v+(1-self.momentum)*pas(t)*grad
            w=w+v
        self.w=w
        self.coef_=w[:-1]

    def predict_proba(self,xtest):
        X=np.hstack([np.array(xtest),np.ones(len(xtest)).reshape(len(xtest),1)])
        pred=np.array([self.sigmoid(X[i].dot(self.w)) for i in range(len(X))])
        return pred
    def predict(self,xtest):

        X=np.hstack([np.array(xtest),np.ones(len(xtest)).reshape(len(xtest),1)])
        pred=self.w.dot(X.T)
        return pred
    def score(self,x,y):
        return  np.mean((self.predict(x)-y)**2)
    
    
class huber:
    """ Class for computation of huber's estimator.
    """
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
