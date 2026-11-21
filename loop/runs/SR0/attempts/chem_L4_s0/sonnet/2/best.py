def equation(t, A, params):
    return (params[0]+params[1]*A)*(1-np.exp(-params[2]*A*t)) - params[3]*A*(1-np.exp(-params[4]*t)) - params[5]/A**params[6]