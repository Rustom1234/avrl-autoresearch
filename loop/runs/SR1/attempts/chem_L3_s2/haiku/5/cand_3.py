def equation(t, A, params):
    a,b,c,d,e,f=params[0],params[1],params[2],params[3],params[4],params[5]
    return (a + b*A/(c+A))*(1-np.exp(-d*t)) + e*t*np.exp(-f*t)
