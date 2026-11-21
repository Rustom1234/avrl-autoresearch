def equation(t,x,v,params):
    return params[0]*x**params[1]+params[2]*v**params[3]+params[4]*np.cos(params[5]*t+params[6])*np.log(v)+params[8]+params[9]*t+params[7]*np.cos(params[5]*t+params[6])