def equation(t, A, params):
    return -params[0]*np.exp(-params[1]*t)*(A-params[2]) + params[3]*np.exp(-params[4]*t)*(A-params[5])**2*0 + params[6]*np.exp(-params[7]*t)*A**2