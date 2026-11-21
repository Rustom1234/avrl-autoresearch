def equation(t,P,params):
    return (params[0]+params[1]*t+params[2]*t**2+params[3]*P+params[4]*P**2
            +params[5]*t*P+params[6]*t**2*P+params[7]*t*P**2+params[8]*t**3*P+params[9]*t**3)
