def equation(t, A, params):
    return params[0]*A**3/t + params[1]*A**1.5 + params[2]*t*np.log(np.abs(params[3])*t) + params[4]