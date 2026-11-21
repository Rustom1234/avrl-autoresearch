def equation(t, A, params):
    return -(params[0]+params[1]*t)*A*np.exp(-params[2]*A)
