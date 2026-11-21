def equation(t, P, params):
    a,b,c,d,e=params[0],params[1],params[2],params[3],params[4]
    return -a*(1-np.exp(-b*t))*(1+c*P+d*P**2) + e*P*0
