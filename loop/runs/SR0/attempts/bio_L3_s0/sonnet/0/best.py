def equation(t,P,params):
    s=t**params[0]
    return params[1]/P+params[2]*P**-0.5+params[3]+params[4]*P+params[5]*P**2+s*(params[6]+params[7]/P+params[8]*P+params[9]*P**-0.5)