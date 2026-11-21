def equation(t,x,v,params):
    return params[0]/v**params[5]+params[1]+params[2]*t**params[3]+params[4]*np.sin(params[6]*t)