def equation(t, A, params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return (a*A + b*A**2)*(1-np.exp(-c*t)) + d*A*np.exp(-e*t) + f*A**2*np.exp(-g*t)
