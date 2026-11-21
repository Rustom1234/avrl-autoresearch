def equation(t, P, params):
    r,K,c,d,e = params[0],params[1],params[2],params[3],params[4]
    return r*P*(1-P/K) - c/P + d*P**2 + e*t
