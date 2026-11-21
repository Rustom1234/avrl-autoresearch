def equation(t,P,params):
    return (params[0]+params[1]*P**params[2]+params[3]/P**params[4])*(1+params[5]*t)