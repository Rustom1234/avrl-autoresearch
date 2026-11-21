def equation(t,P,params):
    r,A,K,H=params[0],params[1],params[2],params[3]
    return r*P*(P/A-1)*(1-P/K)-H