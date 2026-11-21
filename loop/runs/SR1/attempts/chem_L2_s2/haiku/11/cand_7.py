def equation(t, A, params):
    return -params[0]*A/(params[1]+A)*np.exp(-params[2]*t) + params[3]*A*A - params[4]*np.exp(-params[5]*t) + params[6]*A*np.exp(-params[7]*t)