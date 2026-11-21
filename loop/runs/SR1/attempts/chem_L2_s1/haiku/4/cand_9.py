def equation(t, A, params):
    return -(params[0]*A**params[1]*np.exp(params[2]*t))/(1+params[3]*A*t)
