def equation(t, P, params):
    return params[0] + params[1]*t + params[2]*P + params[3]*t*P + params[4]*P*P + params[5]*t*t + params[6]*t*t*P + params[7]*t*P*P + params[8]/P + params[9]*t/P
