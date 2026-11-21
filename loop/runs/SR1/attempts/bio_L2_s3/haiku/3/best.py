def equation(t, P, params):
    return (params[0]+params[1]*t+params[2]*P+params[3]*t*P+params[4]*P**2
            + params[5]*np.exp(params[6]*t) + params[7]*np.exp(params[6]*t)*P + params[8]*t**2)