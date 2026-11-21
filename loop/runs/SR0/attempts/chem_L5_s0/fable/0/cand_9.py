def equation(t, A, params):
    return params[0]+params[1]*A+params[2]*A**params[3]/(1+np.abs(params[4])*A**(-np.abs(params[5])))