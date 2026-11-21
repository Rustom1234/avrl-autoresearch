def equation(t,P,params):
    return (params[0]+params[1]*np.sin(params[2]*t+params[3]))*np.log(P)+params[4]+params[5]*t+params[6]*np.sin(params[2]*t+params[7])