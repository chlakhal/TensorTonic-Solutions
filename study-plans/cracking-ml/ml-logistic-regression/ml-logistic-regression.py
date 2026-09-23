import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X=np.array(X,dtype=np.float64)
    y=np.array(y,dtype=np.float64)
    n=len(y)
    ##initialize b and W
    W=np.zeros(X.shape[1],dtype=np.float64)
    b=0.0
    for _ in range(n_iters):
        y_pred=1/(1+np.exp(-(X@W+b)))
        error=y_pred-y 
        dW=(1/n)*(X.T@error)
        db=(1/n)*(np.sum(error))
        W-=lr*dW
        b-=lr*db
    return np.round(W,4).tolist(),np.round(float(b),4)
    
