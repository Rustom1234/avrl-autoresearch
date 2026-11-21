def equation(t,P,params):
    a,K,H,c,d=params[0],params[1],params[2],params[3],params[4]
    return a*P*(1-P/K) - H*t/(t+c) + d*P*t/(t+c)
