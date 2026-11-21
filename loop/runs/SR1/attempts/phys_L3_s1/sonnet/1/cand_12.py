def equation(t,x,v,params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]*np.log(v) + v*(params[4]+params[5]*t+params[6]*t**2+params[7]*t**3) + params[8]+params[9]*x