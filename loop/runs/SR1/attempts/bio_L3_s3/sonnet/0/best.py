def equation(t,P,params):
    e=np.exp(-params[8]*t)
    return params[0]+params[1]*P+params[2]*np.sqrt(P)+e*(params[3]+params[4]*P)+params[5]*t*P+params[6]*t+params[7]*t*np.sqrt(P)+params[9]*t*t