def equation(t, P, params):
    return P*(params[0]+params[1]*t+params[2]*t**2) + params[3]*P**2 + params[4]*np.sin(params[5]*P+params[6]) + params[7]*P*t