def equation(t,x,v,params):
    w=params[1]
    return params[0]*np.sin(w*t+params[2])+params[3]+params[6]*np.sin(2*w*t+params[7])+params[4]*np.sin(3*w*t+params[5])