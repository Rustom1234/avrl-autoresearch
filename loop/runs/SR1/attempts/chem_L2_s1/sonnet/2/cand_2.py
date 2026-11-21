def equation(t, A, params):
    return -params[0]*np.log(1+params[1]*t)*A**params[2]