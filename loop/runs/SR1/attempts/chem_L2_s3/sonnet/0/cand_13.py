def equation(t, A, params):
    return -(params[0] + params[1]*np.exp(-params[2]*t) + np.sqrt(t)*(params[3]+params[5]*A**2))