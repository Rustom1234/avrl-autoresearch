def equation(eps, T, params):
    return params[0]*eps**2/(1+(0.1*params[1])**2*eps**2) + 0.01*params[2]*T**3 + 0.001*params[3]*eps**3*T**2