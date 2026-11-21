def equation(t, P, params):
    return params[0]/(P+params[1]) + params[2]*P + (params[3]+params[4]*t)*P*(1-P/params[5])+params[7]+params[6]*P**2+params[8]*P**3*(1+params[9]*t)