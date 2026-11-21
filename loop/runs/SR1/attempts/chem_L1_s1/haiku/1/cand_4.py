def equation(t, A, params):
    return params[0]*np.log(params[1]/A) + params[2]*(params[3]-A)
