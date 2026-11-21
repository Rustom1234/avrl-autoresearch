def equation(t,P,params):
    return params[0]/P + params[1]*P + params[5] + params[2]*t*P*np.exp(-params[3]*P**2)+params[4]*t