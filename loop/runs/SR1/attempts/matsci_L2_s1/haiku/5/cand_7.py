def equation(eps, T, params):
    a,b,c,d,e,f,g = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return -(a + b*np.power(eps, c) + d*T + e*T**2 + f*eps*T + g*eps*T**2)
