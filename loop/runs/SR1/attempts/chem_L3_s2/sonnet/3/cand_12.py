def equation(t, A, params):
    return params[0]*np.log(1+params[1]*t)*np.exp(-params[2]*t) + params[3]*np.exp(-params[4]*t)*A**3+params[5]