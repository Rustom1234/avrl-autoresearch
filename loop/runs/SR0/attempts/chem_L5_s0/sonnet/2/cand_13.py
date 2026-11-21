def equation(t,A,params):
    return params[0]*A**params[1]/(1+params[2]*A**params[3])+params[4]*A+params[5]*A**2+params[6]*A**3