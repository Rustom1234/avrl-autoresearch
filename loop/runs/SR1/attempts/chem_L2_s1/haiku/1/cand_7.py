def equation(t, A, params):
    return -(params[0]*A*np.exp(-params[1]*t)/(1+params[2]*A) + params[3]*A*np.exp(params[4]*t)/(1+params[5]*A))