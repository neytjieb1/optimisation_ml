import sys
import numpy as np
from numpy import round

def dot(x, W):
    value = np.dot(W, x)

    def vjp(u):
        vjp_wrt_W = np.outer(u, x)  #applied to W
        vjp_wrt_x = W.T.dot(u)  #applied to x
        return vjp_wrt_x, vjp_wrt_W
        
    return value, vjp

def relu(x):
    value = np.maximum(0, x)

    def vjp(u):
        gdash = lambda y: 1 if y>=0 else 0
        vjp_wrt_x = u*np.vectorize(gdash)(x)
        return vjp_wrt_x,  
        # The comma is important!
    
    return value, vjp

def initialiseMLP_random(inputfeatures, layers, unif=False, verbose=False):
    dims = np.random.choice([i for i in range(2,8)], layers)
    if unif:
        W = [np.random.uniform(-1, 1, size=(dims[0], inputfeatures))]
    else:
        W = [np.array(np.random.rand(dims[0], inputfeatures))]
    for i in range(1, len(dims)):
        if unif:
            Wi = np.random.uniform(-1, 1, size=(dims[i], dims[i-1]))
        else:
            Wi = np.array(np.random.rand(dims[i], dims[i-1]))
        W.append(Wi)

    W.reverse()
    if unif:
        x = np.random.uniform(-1, 1, inputfeatures)
        u = np.random.uniform(-1, 1, dims[-1])
    else:
        x = np.random.uniform(0, 1, inputfeatures)
        u = np.random.uniform(0, 1, dims[-1])

    if verbose:
        print("u=", np.shape(u))
        for i in range(len(W)):
            print("W{i}=".format(i=i), np.shape(W[i]))
        print("x=", np.shape(x))

    return x, W, u

def mlp2(x, W): # W = [Wk, ..., W3, W2, W1]
    """
    input: 
        x = input data
        W1 = weight matrix
        W2 = weight matrix
    formula:
        y = W2.q(W1.x)
    returns:
        value = evaluated value according to formula
        vjp = tuple of vjp's in order d/dx, d/dW1, d/dW2

    """
    x1 = x
    x2, d2 = dot(x1, W[1])     #f1
    x3, d3 = relu(x2)         #f2
    x4, d4 = dot(x3, W[0])    #f3
    
    def vjp(u):
        jf_W = []
        jf4_x, jf_Wk = d4(u)
        jf3_x,       = d3(jf4_x)
        jf2_x, jf_W  = d2(jf3_x)
        vjp_x = jf2_x#[jf4_x, jf3_x, jf2_x]
        vjp_W = [jf_Wk, jf_W]
        return vjp_x, vjp_W

    o = x4
    return o, vjp

def mlpk(x, W): #W = [Wk, ..., W3, W2, W1]
    x1, d1 = mlp2(x, W[-2:])
    if len(W)==2:
        return x1, d1
    xi = [x, x1]
    di = [-1, d1]
    for k in range(3, len(W)+1): # print(k)
        xk, dk = relu(xi[-1])
        xi.append(xk)
        di.append(dk)
        xk, dk = dot(xi[-1], W[-k])
        xi.append(xk)
        di.append(dk)
    o = xi[-1] #last value
    
    xi = xi[1:]
    di = di[1:]

    def vjp(u):
        jf_x = [u]
        jf_W = []
        for k in range(1, len(di), 2):        #di = [mlp2, relu, W3, relu, W4]
            # print("k=", k)                          #xi = [mlp2, relu, W3, relu, W4]
            jfk_x, jfk_W = di[-k](jf_x[-1])
            jf_x.append(jfk_x)
            jf_W.append(jfk_W)
            jfk_x, = di[-(k+1)](jf_x[-1])
            jf_x.append(jfk_x)
        
        jfk_x, jfk_W = di[0](jf_x[-1])      #This will be mlp2
        jf_x.append(jfk_x)
        jf_x = jf_x[1: ]                    #remove u which was added at start
        jf_W.extend(jfk_W)

        return jf_x[-1], jf_W            

    return o, vjp


#Note: W is passed back in the form: W = [Wk, ..., W3, W2, W1] 
if __name__ == '__main__':
    np.random.seed(62394)
    #Run checks from 2-10 layers
    l = 10
    #Each layer has arbitrary # input features
    inputfeatures = np.random.choice([i for i in range(2,11)], l+1)
    
    verbose = True  #Full outputs
    negintv = False     #if true, generated on interval U[-1, 1] else generated on U[0,1]
    fname = 'outputB_neg.txt' if negintv else 'outputB_nonneg.txt'


    with open(fname, 'w') as sys.stdout:

        print("======================================================================")
        print("                          I START HERE")
        print("======================================================================")
        for i in range(2, l+1):
            print("\n\n========================= LAYERS = {l} ========================".format(l=i))
            print("--------------------- #input features = {f} -----------------------".format(f=inputfeatures[i]))
            x, W, u = initialiseMLP_random(inputfeatures[i], i, negintv, verbose)
            #Let's see if we get an error
            val, vjp = mlpk(x, W)
            #This is where mine often broke, applying u to vjp
            vjp_x, vjp_W = vjp(u)
            # print("\nlen(vjp_wrt_W) =", len(vjp_W))
            
            #Own checks, this might be different for you
            assert(len(vjp_W)==len(W))
            print("Pass")  #Each W has associated derivative

            if verbose:
                print("\n----------------------- vjp_wrt_x -----------------------")
                print(vjp_x)
                print("\n----------------------- vjp_wrt_W -----------------------")
                for w in vjp_W:
                    print(w)

        print("======================================================================")
        print("\tII LET's DO SOME MORE TESTING: RECURSION")
        print("======================================================================")
        #Reset seed value
        np.random.seed(623945)
        x, W, u = initialiseMLP_random(4, 3, False, True)

        def f_rec(x,W):
            if len(W)==1:
                temp = np.dot(W, x).reshape(-1)
                return temp
            temp = np.dot(W[0], np.maximum(f_rec(x, W[1:]), 0))
            return temp

        val = np.array(np.dot(W[0], np.maximum(0, np.dot(W[1],np.maximum(np.dot(W[2], x), 0) ) )))
        if verbose:
            print(val)
            print(f_rec(x,W))

        assert(np.round(f_rec(x,W),6) == np.round(val,6)).all(), "Recursive f not working"
        print("Pass: recursive construction")


        print("\n======================================================================")
        print("\tIII THIS IS GETTING FUN: FINITE DIFFERENCES")
        print("======================================================================")
        #Reset seed value
        np.random.seed(623945)
        verbose = True
        negintv = False

        #Small matrix
        x, W, u = initialiseMLP_random(4, 5, negintv, verbose)
        print("Input: ", x.shape, "\nOutput: ", u.shape, "\nJacobian: ({n}, {m})\n".format(n=len(u), m=len(x) ))

        val, vjp = mlpk(x, W)
        # f = lambda x, W2=W[0], W1=W[1]: np.dot(W2, np.maximum(0, np.dot(W1, x)))
        
        sign_dig = 6
        assert((round(f_rec(x, W), sign_dig) == round(val, sign_dig)).all()), 'values not equal'
        print("Pass: Equal function values")

        # ei = np.eye(1,len(u),i).reshape(-1)

        #VJP with ei extracts ith row
        calculated_vjpX = []
        calculated_vjpW = []
        for i in range(len(u)):
            vjpxi, vjp_Wi = vjp(np.eye(1,len(u),i).reshape(-1))
            calculated_vjpX.append(vjpxi)
            calculated_vjpW.append(vjp_Wi)
        calculated_vjpX = np.array(calculated_vjpX)
        if verbose: 
            print("\nCalculated JVP values x")
            print(calculated_vjpX)

        # FIN DIFF
        eps = 1e-8
        g = lambda x, eps, e: ( (f_rec(x+eps*e, W) - f_rec(x, W)) / eps )
        
        fin_diff = []
        for i in range(0, len(x)):
            # fin_diff[:,i] = g(x, eps, np.eye(1,len(x),i).reshape(-1))
            fin_diff.append(g(x, eps, np.eye(1,len(x),i).reshape(-1)))
            # print(g(x, eps, np.eye(1,len(x),i).reshape(-1)).shape)
        fin_diff = np.array(fin_diff).T
        if verbose:
            print("Finite differences")
            print(fin_diff)

        assert((round(fin_diff, sign_dig)==round(calculated_vjpX, sign_dig)).all()), "vjp's are not equal"
        print("Pass: Equal VJP's")
