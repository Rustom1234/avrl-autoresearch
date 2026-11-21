def equation(t, A, params):
    return params[0]*np.sqrt(A) + params[1]*np.log(t) + params[2]*t + params[3]*t**2 + params[4] + params[5]*np.exp(-params[6]*t)
