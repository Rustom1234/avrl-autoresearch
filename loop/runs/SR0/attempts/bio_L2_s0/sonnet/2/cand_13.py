def equation(t,P,params):
    return P*(params[0]*np.exp(-params[1]*t)-params[2]*P) + params[3] + params[4]*np.log(t) + params[5]*t + params[6]*np.log(P)*np.exp(-params[7]*t)