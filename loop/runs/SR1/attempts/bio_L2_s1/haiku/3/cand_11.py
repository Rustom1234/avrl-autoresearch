def equation(t, P, params):
    return (params[0] + params[1]*t)*(1 - (P/params[2])**params[3]) + params[4]*t*P