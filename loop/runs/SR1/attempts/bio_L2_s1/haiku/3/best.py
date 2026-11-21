def equation(t, P, params):
    return (params[0] + params[1]*t)*np.exp(-params[2]*P) + params[3]*t*P + params[4]*P**2 + params[5]*P + params[6]