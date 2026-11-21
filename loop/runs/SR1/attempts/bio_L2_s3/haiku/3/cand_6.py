def equation(t, P, params):
    K=params[0]
    return (K-P)*(params[1]+params[2]*t+params[3]*P+params[4]*t*P+params[5]*t**2)