def importData(p=0.2):
    from sklearn import datasets
    dataset = datasets.fetch_california_housing(as_frame = True)
    cols = dataset.frame.columns
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    np.random.seed(1)

    dataset.frame_normalized = StandardScaler().fit_transform(dataset.frame)
    # We drop Longitude as well since Latitude has enough information
    X = dataset.frame_normalized[:,0:-1]
    y = dataset.frame_normalized[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p, random_state = 16)
    X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

    return X_train, y_train, X_test, y_test, cols

from scipy.linalg import norm
import numpy as np

#Part III
class RegPb(object):
    '''
        A class for regression problems with linear models.
        
        Attributes:
            A: Data matrix (features)
            y: Data vector (labels)
            n,d: Dimensions of A
            loss: Loss function to be considered in the regression
                'l2': Least-squares loss
                'logit': Logistic loss
            lbda: Regularization parameter
    '''
   
    # Instantiate the class
    def __init__(self, A, y,lbda=0,loss='l2'):
        self.A = A
        self.y = y
        self.n, self.d = A.shape
        self.loss = loss
        self.lbda = lbda
        
    # Objective value
    def fun(self, x):
        if self.loss=='l2':
            return norm(self.A.dot(x) - self.y) ** 2 / (2. * self.n) + self.lbda * norm(x) ** 2 / 2.
        elif self.loss=='logit':
            yAx = self.y * self.A.dot(x)
            return np.mean(np.log(1. + np.exp(-yAx))) + self.lbda * norm(x) ** 2 / 2.
    
    # Partial objective value
    def f_i(self, i, x):
        if self.loss=='l2':
            return norm(self.A[i].dot(x) - self.y[i]) ** 2 / (2.) + self.lbda * norm(x) ** 2 / 2.
        elif self.loss=='logit':
            yAxi = self.y[i] * np.dot(self.A[i], x)
            return np.log(1. + np.exp(- yAxi)) + self.lbda * norm(x) ** 2 / 2.
    
    # Full gradient computation
    def grad(self, x):
        if self.loss=='l2':
            return self.A.T.dot(self.A.dot(x) - self.y) / self.n + self.lbda * x
        elif self.loss=='logit':
            yAx = self.y * self.A.dot(x)
            aux = 1. / (1. + np.exp(yAx))
            return - (self.A.T).dot(self.y * aux) / self.n + self.lbda * x
    
    # Partial gradient
    def grad_i(self,i,x):
        a_i = self.A[i]
        if self.loss=='l2':
            return (a_i.dot(x) - self.y[i]) * a_i + self.lbda*x
        elif self.loss=='logit':
            grad = - a_i * self.y[i] / (1. + np.exp(self.y[i]* a_i.dot(x)))
            grad += self.lbda * x
            return grad
        
    # Partial gradient knowing the model (useful for certain gradient techniques)
    def grad_ai(self,i,aix,x=None):
        a_i = self.A[i]
        if self.loss=='l2':
            grad = (aix - self.y[i]) * a_i 
            if (self.lbda>0):
                grad += self.lbda*x
        elif self.loss=='logit':
            grad = - a_i * self.y[i] / (1. + np.exp(self.y[i]* aix))
            if (self.lbda>0):
                grad += self.lbda * x
        return grad        

    # Lipschitz constant for the gradient
    def lipgrad(self):
        if self.loss=='l2':
            L = norm(self.A, ord=2) ** 2 / self.n + self.lbda
        elif self.loss=='logit':
            L = norm(self.A, ord=2) ** 2 / (4. * self.n) + self.lbda
        return L
    
    # ''Strong'' convexity constant (could be zero if self.lbda=0)
    def cvxval(self):
        if self.loss=='l2':
            s = svdvals(self.A)
            mu = min(s)**2 / self.n 
            return mu + self.lbda
        elif self.loss=='logit':
            return self.lbda

def stoch_grad(x0,problem,xtarget,stepchoice=0,step0=1, n_iter=1000,nb=1,average=0,scaling=0,with_replace=False,verbose=False): 
    """
        A code for gradient descent with various step choices.
        
        Inputs:
            x0: Initial vector
            problem: Problem structure
                problem.fun() returns the objective function, which is assumed to be a finite sum of functions
                problem.n returns the number of components in the finite sum
                problem.grad_i() returns the gradient of a single component f_i
                problem.lipgrad() returns the Lipschitz constant for the gradient
                problem.cvxval() returns the strong convexity constant
                problem.lambda returns the value of the regularization parameter
            xtarget: Target minimum (unknown in practice!)
            stepchoice: Strategy for computing the stepsize 
                0: Constant step size equal to 1/L
                1: Step size decreasing in 1/sqrt(k+1)
            step0: Initial steplength (only used when stepchoice is not 0)
            n_iter: Number of iterations, used as stopping criterion
            nb: Number of components drawn per iteration/Batch size 
                1: Classical stochastic gradient algorithm (default value)
                problem.n: Classical gradient descent (default value)
            average: Indicates whether the method computes the average of the iterates 
                0: No averaging (default)
                1: With averaging
            scaling: Use a diagonal scaling
                0: No scaling (default)
                1: Average of magnitudes (RMSProp)
                2: Normalization with magnitudes (Adagrad)
            with_replace: Boolean indicating whether components are drawn with or without replacement
                True: Components drawn with replacement
                False: Components drawn without replacement (Default)
            verbose: Boolean indicating whether information should be plot at every iteration (Default: False)
            
        Outputs:
            x_output: Final iterate of the method (or average if average=1)
            objvals: History of function values (Numpy array of length n_iter at most)
            normits: History of distances between iterates and optimum (Numpy array of length n_iter at most)
    """
    ############
    # Initial step: Compute and plot some initial quantities

    # objective history
    objvals = []
    normits = []
    
    # Lipschitz constant
    L = problem.lipgrad()
    
    # Number of samples
    n = problem.n
    
    # Initial value of current iterate  
    d = len(x0)
    x = x0.copy()
    nx = norm(x)
    
    # Average (if needed)
    if average:
            xavg=np.zeros(len(x))
    
    #Scaling values
    if scaling>0:
        mu=1/(2 *(n ** (0.5)))
        v = np.zeros(d)
        beta = 0.8
    
    obj = problem.fun(x) 
    objvals.append(obj);
    nmin = norm(x-xtarget)
    normits.append(nmin)
    
    # if verbose:
    #     # Plot initial quantities of interest
    #     print("Stochastic Gradient, batch size=",nb,"/",n)
    #     print(' | '.join([name.center(8) for name in ["iter", "fval", "normit"]]))
    #     print(' | '.join([("%d" % k).rjust(8),("%.2e" % obj).rjust(8),("%.2e" % nmin).rjust(8)]))
    
    k=0
    e=1
    while (k < n_iter and nx < 10**100):
        # Draw the batch indices
        ik = np.random.choice(n,nb,replace=with_replace)# Batch gradient
        # Stochastic gradient calculation
        sg = np.zeros(d)
        for j in range(nb):
            gi = problem.grad_i(ik[j],x)
            sg = sg + gi
        sg = (1/nb)*sg
        
        if scaling>0:
            if scaling==1:
                # RMSProp update
                v = beta*v + (1-beta)*sg*sg
            elif scaling==2:
                # Adagrad update
                v = v + sg*sg 
            sg = sg/(np.sqrt(v+mu))

        if stepchoice==0:
            x[:] = x - (step0/L) * sg
        elif stepchoice==1:
            sk = float(step0/(np.sqrt(k+1)))
            x[:] = x - sk * sg
        elif stepchoice>0:
            sk = float(step0/((k+1)**stepchoice))
            x[:] = x - sk * sg
        
        nx = norm(x) #Measure divergence 
        
        if average:
            xavg = k/(k+1) *xavg + x/(k+1) 
            nmin = norm(xavg-xtarget)
            obj = problem.fun(xavg)
        else:
            obj = problem.fun(x)
            nmin = norm(x-xtarget)
        
        k += 1
        # Plot quantities of interest at the end of every epoch only
        # if (k*nb) % n == 0:
        if k*nb - e*n >= 0: 
            print("Epoch", e, end='\r')
            e+=1
            objvals.append(obj)
            normits.append(nmin)
    
    # Plot quantities of interest for the last iterate (if needed)
    # if (k*nb) % n > 0:
    objvals.append(obj)
    normits.append(nmin)
    print('')
    
    # Outputs
    if average:
        x_output = xavg.copy()
    else:
        x_output = x.copy()
    
    return x_output, np.array(objvals), np.array(normits)

#Part IV
def project(l, x, R):
  """
  Function to project the vector x onto a convex set
  
  Inputs:
      l: Strategy for projection
        0: project to simplex with sum R
        1: project to l1 ball with radius R
        2: project to l2 ball with radius R
      x: Vector before projection
      R: radius/sum

  Outputs:
      projected vector value
    """
 
  def proj_to_l2ball(x, R):
      temp = np.maximum(R, np.linalg.norm(x, 2))
      return R*x / np.maximum(R, np.linalg.norm(x, 2))

  def proj_to_simplex(x, R):
    if np.sum(x)==R and np.alltrue(x>=0):
      return x
    u = np.sort(x)[::-1]
    cum = np.cumsum(u)
    K = np.nonzero([(cum[k]-R)/(k+1) for k in range(len(x))] < u)[0][-1] + 1
    tau = (cum[K-1]  - R)/K
    xn = np.maximum(x-tau, 0)
    assert(np.allclose(np.sum(xn), R, rtol=1e-4)), 'not properly projected to simplex'
    return xn

  def proj_to_l1(x, R = 5):
    u = np.abs(x)
    if u.sum() <= R:
        return x
    return proj_to_simplex(u, R) * np.sign(x)

  if l==1:
    return proj_to_l1(x, R)
  elif l==2:
    return proj_to_l2ball(x, R)
  else:
    return proj_to_simplex(x, R)

def ProjGD_Ball(th0, F, gradF, step, max_iter, projectionfunction, tol, R=1):
    """
    Function implementing projected gradient descent (PGD) onto the simplex, l1-ball or l2-ball

    Inputs:
        th0: initial point
        F: objective function
        gradF: function computing gradient of the objective
        step: step-size parameter
        max_iter: maximum number of iteration before stopping
        projectionfunction: function computing chosen projection
        tol: tolerance for objective; stopping condition
        R (optional): radius/sum of ball with default set to 1

    Outputs:
        iters: array of iterates generated for theta
        errs: array of successive mean-squared errors. 
    """

    iters =[th0]
    errs = []
    th = th0

    k = 0
    while (True):
        th = projectionfunction(th - step*gradF(th), R)
        # print(np.linalg.norm(th,2), f'<={radius}: ', np.linalg.norm(th, 2)<=radius)
        iters.append(th)
        # errs.append(np.linalg.norm(iters[-1] - iters[-2]))
        errs.append(np.linalg.norm(F(iters[-1])-F(iters[-2])))
        if (k>max_iter or errs[-1]<=tol):
            break
        k = k + 1
        
    return np.array(iters).T, np.array(errs)

def argmin(xk, R, q):
    """
    Function implementing the linear minimisation as in equation (1)
    Inputs:
        xk: Vector point at which minimised
        R: radius/sum parameter depending on convex set
        q: Choice parameter determining convex set:
            0: Simplex
            1: l1-ball
            2: l2-ball
    Outputs:
        argmin of argument (1) 
    """
    def argmin_l2ball(xk, q, R):
        # for lq ball
        p = q/(q-1)
        a = np.zeros_like(xk)
        denum = np.sum(np.abs(xk)**p)**(1/q)
        for j in range(len(xk)):
            a[j] = - R*(np.sign(xk[j])*np.abs(xk[j])**(p-1))/denum        
        return a

    def argmin_l1ball(xk, R):
        d = len(xk)
        k0 = np.argmax(np.abs(xk))
        return -R*np.sign(xk[k0])*np.eye(1,d, k0).reshape(-1)


    if q==1:
        return argmin_l1ball(xk, R)
    else:
        return argmin_l2ball(xk, q, R)

def extrpt(dim, R, q):
    """
    Function to choose a random extreme point for a specific convex set.
    Inputs:
        d: dimension
        R: set-specific parameter (sum/radius)
        q: parameter specifying the which convex set to be considered
            0: Simplex
            1: l1-ball
            2: l2-ball
    Outputs:
        random extreme point for a specific convex set 
    """

    def extremept_l2(dim, q, R):
        # for lq ball
        ext = np.random.randn(dim)
        return R*(ext / np.linalg.norm(ext, q))

    def extremept_l1(dim, R): 
        i = int(np.random.choice(dim, 1))
        return R*np.eye(1,dim, i).reshape(-1)

    if q==1:
        return extremept_l1(dim, R)
    else:
        return extremept_l2(dim, q, R)

def updatetheta(flag, k, sk, f, gradf):
    '''
    Function implementing the weights assigned to convex combination of linearisation.
    Inputs:
        flag: 
            'ls': exact linesearch, see (3)
            'fixed': fixed stepsize, see (4)
        k: iteration at which the combination occurs
        sk: linearisation at iteration k
        f: objective function
        gradf: gradient function for objective
    Outputs:
        thetak: associated weight for update    
    '''
    
    
    def backtracking_ls(sk, f, gradf):
        '''
        Function implementing backtracking linesearch according to formula (3)
        Inputs:
            sk: descent direction
            f: objective function
            gradf: gradient of objective function
        '''

        maxiter = 100
        a = 0.25 #0<a<0.5
        b = 0.5 #0<b<1
        nu = 1
        it = 0
        while True:
            if f(sk - nu*gradf(sk)) >= (f(sk) - a*nu*np.linalg.norm(sk,2)**2):
                # print(f'ls: theta found at iter {it}')
                return nu
            elif it > maxiter:
                print('not found')
                break
            nu = b*nu
            it = it+1
        return 1/0

    if flag=='fixed':
        return 2/(k+2)
    elif flag == 'ls':
        return backtracking_ls(sk, f, gradf)
    else:
        print('Error update rule')
        return

def CondGD(f, gradf, R, pick_theta_rule, extr_pt_rule, argminrule, dim, maxit=500):
    '''
    Function implementing the conditional gradient descent method (also, Franke-Wolfe method)
    Inputs:
        f: objective function
        gradf: gradient of objective function
        R: set-specific parameter (sum/radius)
        pick_theta_rule: function for updating the weight theta
        extr_pt_rule: function for choosing an initial starting point at the extreme point of the associated convex set
        argminrule: function for picking the minimum as in (1)
        dim: dimension of problem
        maxit: (optional), maximum iteration before enforcing stopping
    Output:
        x: an array of iterations of the solution x
        errs: an array of errors determined at each iteration with the squared loss function
    '''
    
    x0 = extr_pt_rule(dim, R)
    k = 0
    x = [x0]
    errs = [1e10]
    while (k<=maxit):
        sk = argminrule(gradf(x[-1]), R)

        if np.dot(gradf(x[k]), sk-x[-1]) >= 0:
            print(f'Success at iter {k}')
            break
        thk = pick_theta_rule(k, sk)
        
        x.append( thk*sk + (1-thk)*x[-1])
        errs.append(np.linalg.norm(x[-1] - x[-2], 2))
        k = k+1
    if k>maxit and False:
        print('Cond: Not found')
    return np.array(x), np.array(errs)

#Part IV

#Part V
SoftThresh = lambda x, tau: np.sign(x) * np.maximum(np.abs(x)-tau, 0.0)
GD_loss = lambda A,x, y: np.linalg.norm(A.dot(x) - y, 2) ** 2 / (2. * A.shape[0]) 
GD_grad = lambda A, x, y: (A.T).dot(A.dot(x) - y)*(1/A.shape[0])

ridge_loss = lambda A, x, y, lbda: GD_loss(A,x,y) + lbda* np.linalg.norm(x,2) ** 2 / 2.
ridge_grad = lambda A, x, y, lmbd: GD_grad(A,x,y) + 1*lmbd*x

def GD(X, y, niter, step=0.001, lmbd=0):
    n, d = X.shape
    L = np.linalg.norm(X, 2) ** 2 / n + lmbd
    step = 1/L
    theta = np.ones(X.shape[1])
    loss_evol = np.zeros(niter)
    for i in range(niter):
        error_k = np.dot(X, theta) - y
        g = ridge_grad(X, theta, y, lmbd)
        theta = theta - (1/L)*g
        loss_evol[i] = ridge_loss(X, theta, y, lmbd)
    return theta, loss_evol

def ISTA(A, y, lbda, step, niter):
    (n, d) = A.shape
    x = np.ones(d)
    
    Func = [GD_loss(A,x, y)]

    for k in range(niter):
        g = GD_grad(A,x, y) #smooth part
        x = SoftThresh(x - step*g, step*lbda)
        Func.append(GD_loss(A,x,y))

    return x, Func

