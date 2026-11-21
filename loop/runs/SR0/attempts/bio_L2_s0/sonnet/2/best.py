def equation(t,P,params):
    return P*(params[0]*np.exp(-params[1]*t)-params[2]*P*np.exp(-params[3]*t)) + params[4] - params[5]*np.tanh(params[6]*t)-params[7]*np.log(t)