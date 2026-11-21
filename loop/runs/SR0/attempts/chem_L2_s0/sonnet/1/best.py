def equation(t, A, params):
    return -params[0]*A**(-params[1])*np.exp(-params[2]*t) - params[3] + params[4]*np.exp(-params[5]*t)*A**params[6]+params[7]*A