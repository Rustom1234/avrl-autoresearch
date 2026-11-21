def equation(t, P, params):
    return P*(params[0] - params[1]*P) + params[2]*t*P + params[3]*t + params[4]*P*P*t + params[5]*t*t*P