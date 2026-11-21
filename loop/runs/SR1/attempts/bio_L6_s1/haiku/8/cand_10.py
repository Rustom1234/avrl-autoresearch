def equation(t, P, params):
    return params[0] + params[1]*t - (params[2]+params[3]*t)*P*(1 - P/(params[4]+params[5]*t))