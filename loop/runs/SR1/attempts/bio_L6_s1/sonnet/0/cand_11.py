def equation(t,P,params):
    return params[0]+params[1]*P+params[2]/(P+params[3])+(params[4]+params[5]*P)*t**2+params[6]*t+params[7]*P*t
