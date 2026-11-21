def equation(t,x,v,params):
    return -(params[0]*x**params[1]+params[2]*v**params[3]+params[7])/t**params[4]+params[5]*np.cos(params[6]*t)