def equation(t,P,params):
    return (params[0]+params[1]*t)/(1+params[2]*P**params[3]) + params[4]*P+params[5]*P**2+params[6]*t*P