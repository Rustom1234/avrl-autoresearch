def equation(t,P,params):
    a,b,c,d,k,e=params[0],params[1],params[2],params[3],params[4],params[5]
    g=1-np.exp(-k*t)
    return a*g + b*P + c*P**2 + d*P*g + e*t
