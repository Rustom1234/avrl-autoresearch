def equation(t, A, params):
    return (params[0]+params[2]*A**params[3])*(1-np.exp(-params[1]*t))+params[4]