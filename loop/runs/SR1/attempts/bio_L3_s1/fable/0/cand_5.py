def equation(t, P, params):
    return params[0]/P + params[2]*P*(1+params[3]*np.sin(params[4]*t+params[5]))*(1-P/params[6]) + params[7]*P+params[8]*P**2