def equation(t,P,params):
    return params[0] + params[1]*t + params[2]*P + params[3]*P**2 + params[4]/P + params[5]*t*P + params[6]*np.exp(-t)+params[7]*P*np.exp(-t)
