def equation(eps, T, params):
    k=params[0]
    A=params[1]+params[2]*T+params[3]*T**2
    B=params[4]+params[5]*T+params[6]*T**2
    return A*np.cos(k*eps)+B
