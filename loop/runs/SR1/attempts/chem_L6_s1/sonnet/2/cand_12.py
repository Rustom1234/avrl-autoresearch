def equation(t, A, params):
    return params[0]*np.sqrt(A)*(1+params[1]*t+params[2]*t*t) + params[3]*A*(1+params[4]*t+params[5]*t*t)+params[6]*(1+params[7]*t+params[8]*t*t)