def equation(t,x,v,params):
    return params[0]+params[1]*x+params[2]*x**2+params[3]*x**3+params[4]*x*v+params[5]*np.log(v)+params[6]*v+params[7]/v+params[8]*np.sin(t)+params[9]*np.cos(t)