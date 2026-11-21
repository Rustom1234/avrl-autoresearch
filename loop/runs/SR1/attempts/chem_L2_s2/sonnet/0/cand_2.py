def equation(t, A, params):
    return -params[0]*A/(params[1]+A**2) + params[2]*A**2 + params[3] + params[4]*np.exp(-params[5]*t)*A