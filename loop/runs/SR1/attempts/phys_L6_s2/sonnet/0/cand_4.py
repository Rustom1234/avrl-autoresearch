def equation(t,x,v,params):
    return params[0]*np.log(t+params[1])+params[2]*x+params[3]*x**3+params[4]*v+params[5]*v**2+params[6]