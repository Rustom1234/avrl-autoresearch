def equation(t, P, params):
    return params[0]*P*(P-params[1])*(1-P/params[2]) + params[3]*t*P + params[4]*t**2 + params[5]*t
