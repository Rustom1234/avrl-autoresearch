def equation(t,P,params):
    L=np.log(P)
    return params[0]+params[1]*t+params[2]*np.sqrt(t)+params[3]*P+params[4]*P**2+params[5]*L+params[6]*t*P+params[7]*np.log(t)+params[8]*t*P**2+params[9]*t*L