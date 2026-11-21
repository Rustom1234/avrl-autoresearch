def equation(t, P, params):
    r,K,A,b=params[0],params[1],params[2],params[3]
    return r*P*(1-P/K) - A*(1-np.exp(-b*t))
