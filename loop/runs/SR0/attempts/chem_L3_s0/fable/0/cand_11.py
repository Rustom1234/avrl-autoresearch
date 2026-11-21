def equation(t, A, params):
    return -params[0] + params[1]*A**params[2]/(np.abs(params[3])+A**params[2]) + params[4]*A**params[5]*np.exp(-np.abs(params[6])*t) + params[7]*A**params[8]