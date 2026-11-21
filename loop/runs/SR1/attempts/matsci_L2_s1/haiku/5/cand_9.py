def equation(eps, T, params):
    a0,a1,b0,b1,n0,n1 = params[0],params[1],params[2],params[3],params[4],params[5]
    return -((a0 + a1*T) + (b0 + b1*T)*np.power(eps, n0 + n1*T))
