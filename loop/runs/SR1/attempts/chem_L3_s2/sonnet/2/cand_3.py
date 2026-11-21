def equation(t, A, params):
    return params[0]*np.sin(params[1]*t+params[2]) + params[3]+ params[4]*A + params[5]*A**2 + params[6]*A*t