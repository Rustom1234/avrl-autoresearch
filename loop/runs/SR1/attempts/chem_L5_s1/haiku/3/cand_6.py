def equation(t, A, params):
    return -(params[0]+params[1]*t)*(1+params[2]*A*np.exp(-params[3]*A))