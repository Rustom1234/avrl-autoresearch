def equation(t,P,params):
    return (params[0]*P+params[1]*P**2)*np.exp(-params[2]*t) + params[3] - params[4]*t**params[5] + params[6]*P