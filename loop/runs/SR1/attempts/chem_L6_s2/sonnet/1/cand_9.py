def equation(t,A,params):
    return params[0]*A**params[1] + params[2]*A**(-params[3]) + params[4]*np.exp(params[5]*t)*A**(-params[6])