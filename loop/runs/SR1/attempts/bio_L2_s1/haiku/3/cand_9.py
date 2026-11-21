def equation(t, P, params):
    return (params[0] + params[1]*t + params[2]*t**2)*(1 - P/params[3]) + params[4]*t*P + params[5]*P**2