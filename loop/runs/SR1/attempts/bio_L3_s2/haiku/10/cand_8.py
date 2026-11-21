def equation(t,P,params):
    return params[0]*P - params[1]*P**2 + params[2]*t**params[3] + params[4]*P*t + params[5]*P*t**2 + params[6]*P**2*t
