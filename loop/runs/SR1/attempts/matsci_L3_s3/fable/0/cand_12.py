def equation(eps, T, params):
    r = 1/(1+params[2]*params[2]*eps)
    return params[0]*np.sqrt(T) + params[1]*0.01*T**3*r + params[3] + params[4]*eps**params[5] + params[6]*r + params[7]*np.log(1+params[8]**2*eps)