def equation(t,P,params):
    return params[0]*P/(params[1]+P) - (params[2]*t+params[3]*t**2)*np.exp(-params[4]*P**2) + params[5]*(t/10)**params[6]*(P>0) - params[7]*t