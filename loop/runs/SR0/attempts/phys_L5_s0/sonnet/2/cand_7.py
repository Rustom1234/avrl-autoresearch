def equation(t,x,v,params):
    return -params[0]*np.sin(params[1]*t+params[2])*(1-np.exp(-params[3]*v)) - params[4]*x + params[5]