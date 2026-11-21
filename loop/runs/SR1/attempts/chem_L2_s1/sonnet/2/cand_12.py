def equation(t, A, params):
    return -params[0]*np.log(1+params[1]*A) + t**params[2]*params[3]*(A-params[4])