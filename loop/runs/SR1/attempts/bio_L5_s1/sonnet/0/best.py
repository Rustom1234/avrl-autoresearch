def equation(t,P,params):
    return params[0]+params[1]*P+params[2]*P*P+params[3]*np.exp(-params[4]*P)