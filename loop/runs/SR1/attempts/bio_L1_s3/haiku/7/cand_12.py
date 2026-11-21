def equation(t, P, params):
    return params[0]*P*(1-(P/params[1])**params[2])