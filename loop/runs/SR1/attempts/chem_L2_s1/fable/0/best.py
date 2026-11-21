def equation(t, A, params):
    return -params[0]*np.log(1+params[1]*A) + t**3*(params[2]*A+params[3])