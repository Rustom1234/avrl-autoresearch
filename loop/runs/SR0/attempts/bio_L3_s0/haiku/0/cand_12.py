def equation(t, P, params):
    return params[0] + params[1]*P + params[2]/P + params[3]*t/P + params[4]*t*P + params[5]*t
