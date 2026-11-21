def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P*np.exp(params[3]*t) + params[4]*P + params[5]*P*P + params[6]*t*P + params[7]*P*P*t + params[8]*t*t
