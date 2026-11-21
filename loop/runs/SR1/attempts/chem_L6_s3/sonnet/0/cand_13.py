def equation(t, A, params):
    return params[0]*np.exp(-params[1]/t**3)*np.exp(-params[2]*t) - params[3]*t**2*A**1.5