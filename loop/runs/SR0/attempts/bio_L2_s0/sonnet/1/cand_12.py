def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*np.tanh(params[3]*t) + params[4]*np.log(P)*np.exp(-params[5]*t)+params[6]*P*np.exp(-params[7]*t)