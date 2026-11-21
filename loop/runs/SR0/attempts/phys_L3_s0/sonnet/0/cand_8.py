def equation(t,x,v,params):
    return (params[0]*x**params[1]+params[2]*x)*(np.log(t)+params[3])+params[4]*np.log(v)+params[5]*v**2*np.sqrt(t)+params[6]