import sys
import numpy as np
from numpy import round

# def dot(x, W):
def dot(W, x):
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

#Insert here
def mlp2(x, W):
    """
    input: 
        x = input data
        W = list of weight matrices, W = [Wk, ..., W3, W2, W1]
    formula:
        y = W2.q(W1.x)
    returns:
        value = evaluated value according to formula
        vjp = tuple of vjp's in order x, W
    """
    W2, W1 = W
    a, vjp_dot1 = dot(W1, x)
    b, vjp_relu = relu(a)
    c, vjp_dot2 = dot(W2, b)
    value = c

    def vjp(u):
        # vjp_wrt_W2, vjp_wrt_b = vjp_dot2(u)
        vjp_wrt_b, vjp_wrt_W2 = vjp_dot2(u)
        vjp_wrt_a, = vjp_relu(vjp_wrt_b)
        # vjp_wrt_W1, vjp_wrt_x = vjp_dot1(vjp_wrt_a) 
        vjp_wrt_x, vjp_wrt_W1 = vjp_dot1(vjp_wrt_a) 

        return vjp_wrt_x, [vjp_wrt_W1, vjp_wrt_W2]
    return value, vjp


def mlpk(x, W): #W = [Wk, ..., W3, W2, W1]
    if (len(W)>=3):
        value, vjp_1 = mlpk(x, W[1:len(W)])
    else:
        # value, vjp_1 = mlp2(x, [W[-2], W[-1]]) 04/01 change to:
        return mlp2(x, [W[-2], W[-1]])
    
    value, vjp_2 = relu(value)
    value, vjp_3 = dot(W[0], value)

    def vjp(u):
        vjp_wrt_x, vjp_wrt_Wk = vjp_3(u)
        vjp_wrt_x, = vjp_2(vjp_wrt_x)
        # vjp_wrt_x_wrtW = vjp_1(vjp_wrt_x) 04/01 change to:
        vjp_wrt_x, *vjp_wrt_W = vjp_1(vjp_wrt_x)
        #04/01 add:
        vjp_wrt_W = vjp_wrt_W[0]
        vjp_wrt_W.append(vjp_wrt_Wk)
        return vjp_wrt_x, vjp_wrt_W
        # return vjp_wrt_x_wrtW, vjp_wrt_Wk 04/01 comment out

    return value, vjp


#Note: W is passed back in the form: W = [Wk, ..., W3, W2, W1] 

if __name__ == '__main__':

    from sklearn import datasets
    dataset = datasets.fetch_california_housing(as_frame = True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    np.random.seed(1)

    dataset.frame_normalized = StandardScaler().fit_transform(dataset.frame)
    # We drop Longitude as well since Latitude has enough information
    X = dataset.frame_normalized[:,0:len(dataset.frame.columns) - 2]
    y = dataset.frame_normalized[:,len(dataset.frame.columns) - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 9)
    X_train = np.insert(X_train, 0, np.ones(X_train.shape[0]), axis=1)
    X_test = np.insert(X_test, 0, np.ones(X_test.shape[0]), axis=1)

    n, d = X_train.shape





if __name__ == '__write__':
    np.random.seed(62394)
    #Run checks from 2-10 layers
    l = 10
    #Each layer has arbitrary # input features
    inputfeatures = np.random.choice([i for i in range(2,11)], l+1)
    
    verbose = True  #Full outputs
    negintv = False     #if true, generated on interval U[-1, 1] else generated on U[0,1]
    fname = 'outputB_negInt.txt' if negintv else 'outputB_nonneg.txt'


    with open(fname, 'w') as sys.stdout:

        print("======================================================================")
        print("                          I. START HERE")
        print("======================================================================")
        for i in range(2, l+1):
            print("\n\n========================= LAYERS = {l} ========================".format(l=i))
            print("--------------------- #input features = {f} -----------------------".format(f=inputfeatures[i]))
            x, W, u = initialiseMLP_random(inputfeatures[i], i, negintv, verbose)
            #Let's see if we get an error
            val, vjp = mlpk(x, W)
            #This is where mine often broke, applying u to vjp
            vjp_x, *vjp_W = vjp(u)
            if len(vjp_W)==1: vjp_W = vjp_W[0]
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
        x, W, u = initialiseMLP_random(4, 2, negintv, verbose)
        print("Input: ", x.shape, "\nOutput: ", u.shape, "\nJacobian: ({n}, {m})\n".format(n=len(u), m=len(x) ))

        val, vjp = mlpk(x, W)
        # f = lambda x, W2=W[0], W1=W[1]: np.dot(W2, np.maximum(0, np.dot(W1, x)))
        
        assert((f_rec(x, W) == val).all()), 'values not equal'
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

        sign_dig = 6
        assert((round(fin_diff, sign_dig)==round(calculated_vjpX, sign_dig)).all()), "vjp's are not equal"
        print("Pass: Equal VJP's")

