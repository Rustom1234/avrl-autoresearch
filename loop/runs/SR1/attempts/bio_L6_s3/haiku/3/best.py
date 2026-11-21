def equation(t, P, params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    x=t*P
    return -a*(1-np.exp(-b*x))*(1+c*P+d*P**2) - e*(1-np.exp(-f*t)) + g*P
