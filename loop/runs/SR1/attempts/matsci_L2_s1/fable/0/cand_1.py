def equation(eps, T, params):
    return (params[0]+params[1]*np.abs(eps)**params[2])*(1-params[3]*(T/params[4])**params[5])+params[6]