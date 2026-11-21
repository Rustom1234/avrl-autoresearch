def equation(t,P,params):
    A=-params[0]*(1-np.exp(-params[1]*t))
    return A + (params[2]+params[3]*t)*P*(1-P/params[4]) + params[5]*np.log(P)*np.exp(-params[6]*t)