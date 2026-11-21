def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*A**2+params[3]*A/(params[4]+A)-params[5]*np.exp(-params[6]*t)/(A+params[7])