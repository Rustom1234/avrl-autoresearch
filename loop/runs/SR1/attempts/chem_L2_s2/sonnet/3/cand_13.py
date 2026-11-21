def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*A**2+params[3]*(1-params[4]*np.exp(-params[5]*t))/(A+params[6])