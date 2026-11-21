def equation(t,P,params):
    return params[0]/P + params[1]*P**params[6] + params[5] + params[2]*t*np.sin(params[3]*P)