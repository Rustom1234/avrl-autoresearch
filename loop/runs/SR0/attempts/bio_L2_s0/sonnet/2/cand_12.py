def equation(t,P,params):
    return params[0]*P*np.exp(-params[1]*t) - params[2]*P*P*(1+params[3]*t) - params[4]*np.tanh(params[5]*t) + params[6]*np.sqrt(t)