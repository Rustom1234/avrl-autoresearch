def equation(eps, T, params):
    return params[0]*np.exp(params[1]*T) + params[2]*np.exp(params[3]*T)*((eps+params[4])**params[5]-params[4]**params[5])