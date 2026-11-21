def equation(t, P, params):
    return P*(params[0]+params[1]*np.sin(params[2]*t+params[3]))*(1-P/params[4]) + params[5]*P*t