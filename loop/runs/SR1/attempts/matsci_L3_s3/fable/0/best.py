def equation(eps, T, params):
    r = 1/(1+params[2]*params[2]*eps)
    return params[0]*np.sqrt(T) + params[1]*0.01*T**3*r + params[3] + (params[6] + params[4]*np.abs(eps)**(1.5+0.1*params[5]))*r