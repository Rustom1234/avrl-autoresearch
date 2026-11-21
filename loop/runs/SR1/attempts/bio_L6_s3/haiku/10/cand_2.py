def equation(t,P,params):
    a,K,b,k,c=params[0],params[1],params[2],params[3],params[4]
    return a*P*(1-P/K) - b*(1-np.exp(-k*t)) - c*P*np.exp(-0*t)
