def equation(eps, T, params):
    a,b,c,d,e,f,g,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return a + b*eps + c/T + d*eps*T**2 + e*T + f*eps*T + g*np.log(T) + h*T**2
