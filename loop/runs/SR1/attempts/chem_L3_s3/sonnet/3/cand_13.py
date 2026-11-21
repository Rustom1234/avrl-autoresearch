def equation(t, A, params):
    return params[0]/t**params[6] + params[1] + params[2]*t - params[3]*A**params[4]*np.log(1+params[5]*t)