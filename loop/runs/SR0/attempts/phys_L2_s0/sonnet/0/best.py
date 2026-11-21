def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**3+params[3]*v+params[4]*np.log(v)+params[5]/t+params[6]*x/t+params[7]*v**2+params[8]*np.sin(params[9]*t)