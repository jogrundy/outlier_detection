import numpy as np
import numpy.linalg as la


def ta_1(n,p, r):
    """
    attempt to generate good test case using sinewaves to make high d spiral/curve
    """
    x = np.linspace(-4*np.pi, 4*np.pi, num=p)
    A = np.zeros((p,r))
    for i in range(r):
        #generate a sequence of numbers varying at different speeds
        A[:,i] = np.sin((x+i)/((i+1))) + np.random.rand(p)/2
    B = np.zeros((n,r))
    x = np.linspace(-2*np.pi, 2*np.pi, num=n)
    for i in range(r):
        #generate a sequence of numbers varying at different speeds
        B[:,i] = np.cos((x+i)/((i+1))) + np.random.rand(n)/2
    L_o = np.dot(A,B.T) #the clean part of M
    return L_o

def ta_2(n,p, r):
    """
    attempt to make good test case using random high d path generator
    start with rv, add random noise to rv to simulate high D path for A and B
    then do dot product for lower rank L
    """
    L_o = np.zeros((p,n))

    A = np.zeros((p,r))
    rv = np.random.rand(p)-0.5
    for i in range(r):
        rv = rv+ (np.random.rand(p)-.5)*0.2
        A[:,i] = rv*(r/n)

    B = np.zeros((n,r))
    rv = np.random.rand(n)-0.5
    for i in range(r):
        rv = rv+ (np.random.rand(n)-.5)*0.2
        B[:,i] = rv*(r/p)

    L_o = np.dot(A, B.T)

    return L_o


def ta_3(n,p, r):
    """
    attempt to make good test case using random high d path generator
    start with rv, add random noise to rv to simulate high D path
    will have some lower dimensional structure due to random walk, but
    not actually a proper lower d structure
    """
    L_o = np.zeros((p,n))
    rv = np.random.rand(p)-0.5
    for i in range(n):
        rv = rv+ (np.random.rand(p)-.5)*0.2
        L_o[:,i] = rv
    return L_o

def ta_4(n,p,r):
    """
    simple data generator. a sine wave with a bit of noise. replicated in p dimensions.
    """
    ns = 8/100
    x = np.arange(n)*ns*np.pi
    # x = x+(np.random.randn(n)-0.5)/3
    # x = np.sin(x)
    lst = []
    for i in range(p):
        x_n = x+(np.random.randn(n)-0.5)*0.2
        x_n = np.sin(x_n)
        lst.append(x_n)# + np.random.randn(n)/10)

    return np.array(lst)


def ta_5(n,p,r):
    """
    simple data generator, modulated sinewave, varying sinewave frequency in steps
    with probability about 0.125 every full wave, i.e changing approximately every 8 waves
    """
    ns = int(8*n/100) #gives the number of waves in 400 data points.
    m = 1
    data = []
    n_vec = []
    x = 0
    s = np.random.rand(n)
    st_noise = np.random.rand(p)
    for i in range(n):
        if np.random.rand()<(0.1):
            m = m*(np.random.rand()+0.5)**2
        x = x+m+s[i]
        d = x*ns*np.pi/n
        data.append(d)

    data = np.array(data)
    x = np.sin(data) # sinewave.
    # print(x.shape)
    sn = np.sin(np.arange(p)*np.pi)*0.1-0.05
    lst = []
    # for i in range(p):
    ov = np.outer(np.ones(p), x)
    arr = ov + np.outer(sn, np.ones(n))
    # lst.append(sn + ov)

    # arr = np.array(lst)
    # print(arr.shape)
    return arr


def ta_6(n,p,r, pow=10):
    """
    generate mackey glass series using parameters tau = 30,
    approximated using rk4
    """
    a = 0.2
    b = 0.1
    tau = 30
    x0 = np.random.rand(p) + 0.9
    deltat = 6 #sampling rate, deltat
    history_length = int(tau/deltat)

    def mg(x_t, x_t_minus_tau, a, b,pow):
        x_dot = -b*x_t+ a*x_t_minus_tau/(1+x_t_minus_tau**pow)
        return x_dot

    def mg_rk4(x_t, x_t_minus_tau, deltat, a, b, pow):
        k1 = deltat*mg(x_t,          x_t_minus_tau, a, b, pow)
        k2 = deltat*mg(x_t+0.5*k1,   x_t_minus_tau, a, b, pow)
        k3 = deltat*mg(x_t+0.5*k2,   x_t_minus_tau, a, b, pow)
        k4 = deltat*mg(x_t+k3,       x_t_minus_tau, a, b, pow)
        x_t_plus_deltat = (x_t + k1/6 + k2/3 + k3/3 + k4/6)
        return x_t_plus_deltat

    X = []
    x_t = x0
    for i in range(n):
        X.append(x_t)
        #T.append(time)
        if i > history_length:
            x_t_minus_tau = X[-history_length]
        else:
            x_t_minus_tau = 0
        x_t_plus_deltat = mg_rk4(x_t, x_t_minus_tau, deltat, a, b, pow)
        x_t = x_t_plus_deltat
    X = np.array(X)
    # for i in range(n):
    #     X[i,:] = X[i,:]+ (np.random.randn(p)-0.5)/10

    return X.T

def normalise(X):
    """
    here I mean make max 1 and min 0 and fit everything else in.
    """
    X = (X - np.max(X)) / (np.max(X) - np.min(X))
    X = X+1

    return X

def get_outlier(type, L, out_ind):
    """
    takes in base dataset and generates an outlying datapoint, based on type given
    'point': produces random vector with random sign
    'context': produces a vector from a different part of the time series.
                should be a certain amount distant.
    'stutter': produces a certain number of repeats of the same vector
    """
    p,n = L.shape
    if type == 'point':
        rv = (np.random.rand(p)+1)/2 #make random number between 0.5 and 1
        rv = rv * np.sign(np.random.rand(p)-0.5) #make randomly pos or neg.
    elif type == 'context':
        #chose random n to copy.
        if out_ind  <n/4:
            n_copy = n-int(np.ceil(np.random.rand()*n*3/4) )-1
        elif out_ind > 3*n/4:
            n_copy = int(np.ceil(np.random.rand()*n*3/4) ) -1
        # elif out_ind  or out_ind:
        #     n_copy = n - out_ind
        else:
            n_copy = int(out_ind + n*(np.random.rand()-0.5)/2)
        # print(out_ind, n_copy)
        rv = L[:, n_copy]
    elif type == 'stutter':
        rv = L[:,out_ind]
    else:
        raise
    return rv


def generate_test(n,p,r, p_frac, p_quant,gamma, noise=0.1, ta=1, nz_cols=None, outlier_type='point'):
    """
    generates test case of multidimensional manifold. outliers to be seeded
    as a fraction (gamma*n) of outliers perturbed by a certain fraction of
    parameters (p_frac) by a certain amount (p_quant)
    keeps the same features perturbed for each outlier column
    returns the data set, and indices of outliers
    """
    ta_lst = [ta_1, ta_2, ta_3, ta_4, ta_5, ta_6]
    ta_fn = ta_lst[ta-1]
    # Get base data
    L_o = ta_fn(n,p, r)
    # Add noise
    # print(L_o.shape, ta)
    # for i in range(n):
    #
    #     L_o[:,i] = L_o[:,i] +  (np.random.rand(p)-0.5)*noise
    # Normalise
    L_o = normalise(L_o)
    for i in range(n):

        L_o[:,i] = L_o[:,i] +  (np.random.rand(p)-0.5)*noise
    # chose at random columns to be outliers
    n_outs = max(int(gamma*n), 1)
    n_feats = max(int(p_frac*p), 1)
    if not nz_cols:
        nz_cols = np.random.choice(np.arange(n), size=(n_outs), replace=False)
    elif nz_cols == 'last half':
        nz_cols = np.random.choice(np.arange(int(n/2), n), size=(n_outs), replace=False)
    elif nz_cols == 'first_qu':
        nz_cols = np.random.choice(np.arange(0, int(n/4)), size=(n_outs), replace=False)
    else:
        print('oops')
        raise

    #chose certain random features to be perturbed, generates mask
    perturb_feats = np.random.choice(np.arange(p), size=n_feats, replace=False)
    mask = np.zeros(p)
    mask[perturb_feats]=1
    to_add = np.array([])

    #start working on the main data matrix M
    M_o = np.copy(L_o)

    for col in nz_cols:
        rv = get_outlier(outlier_type, L_o, col)
        if outlier_type == 'stutter':

            snum = int(np.ceil(np.log10(n)))+1
            # print(n,snum)
            for i in range(1, snum):
                M_o[:,(col-snum)+i] = L_o[:,col] #+ rv*p_quant*mask
                to_add = np.append(to_add, int((col-snum)+i) )

        elif outlier_type=='context':
            M_o[:,col] = rv
        else:
            M_o[:,col] = L_o[:,col] + rv*p_quant*mask
    nz_cols = np.append(nz_cols, to_add)
    nz_cols = nz_cols.astype(int)
    # print(nz_cols)
    return M_o.T, nz_cols

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 200
    p = 1
    r = 20
    gamma = 0.05
    p_frac = 0.3
    p_quant = 0.3
    noise = 1
    ta = 3
    nz_cols = 'first_qu'
    data, outs = generate_test(n,p,r, p_frac, p_quant,gamma,noise=0,ta=ta,nz_cols=nz_cols)
    x = np.arange(n)

    plt.plot(x,data, '.')
    plt.plot(outs, data[outs], 'ro')
    plt.show()
