def equation(t, P, params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    return P*(a+b*t+c*t**2) + d*P**2 + e*t*P**2 + f*t*P*0
