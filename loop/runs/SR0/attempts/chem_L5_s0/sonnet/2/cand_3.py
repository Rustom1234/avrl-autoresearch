def equation(t,A,params):
    return params[0]*A/(1+params[1]*A)+params[2]*A**2+params[3]*A**3+params[4]*A**4+params[5]