def equation(t, P, params):
    r = params[0] + params[1]*np.log(t) + params[2]*t
    return r*P**params[3] - params[4]*P**params[5]*(params[6]+params[7]*np.log(t))