def equation(t, P, params):
    return (params[0] + params[1]*t)*P*(1 - P/params[2]) + params[3]*t + params[4]