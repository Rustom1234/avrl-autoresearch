def equation(t, P, params):
    return params[0] + params[1]*P + params[2]*P*P + params[3]/P + t*(params[4]*P + params[5]*P*P) + params[6]*t*P*P*0 + params[7]*P**3
