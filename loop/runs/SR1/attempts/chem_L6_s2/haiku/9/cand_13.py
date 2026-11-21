def equation(t, A, params):
    return -params[0]*A**2*np.exp(-params[1]*t) + params[2]*t**2/A + params[3]*A + params[4]*t*A**2 + params[5]/A**2*np.exp(-params[6]*t) + params[7]*t + params[8]