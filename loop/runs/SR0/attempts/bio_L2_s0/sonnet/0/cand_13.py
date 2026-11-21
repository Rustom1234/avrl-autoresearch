def equation(t,P,params):
    return (params[0]*P+params[1]*P**2)*np.exp(-params[2]*t) + params[3] + params[4]*np.log(t) + params[5]*P + params[6]*np.log(P)