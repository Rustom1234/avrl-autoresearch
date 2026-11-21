def equation(t, A, params):
    return -(params[0]*A**params[1]*(1+params[2]*t**params[6]) + params[3]*np.exp(-params[4]*t)*A**params[7]+params[5])