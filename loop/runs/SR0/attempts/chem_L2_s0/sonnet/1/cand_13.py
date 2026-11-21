def equation(t, A, params):
    return -params[0]*A**(-params[1])*(t+params[2])**(-params[3]) + params[4]*A**params[5]*np.exp(-params[6]*t)