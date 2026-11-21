def equation(t,P,params):
    e=np.exp(-params[4]*P)
    return params[0]+params[1]*P+params[2]*P**2+params[3]*e+params[5]*P*e