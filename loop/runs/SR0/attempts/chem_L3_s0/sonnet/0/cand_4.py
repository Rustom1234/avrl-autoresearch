def equation(t, A, params):
    return -params[0]*np.exp(-params[1]*A)*0 + params[2]*A**params[3]*(1+params[4]*np.sin(params[5]*t+params[6])) - params[0]