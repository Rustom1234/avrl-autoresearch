def equation(t,x,v,params):
    return -params[0]*np.abs(x)**params[1]-params[2]*np.abs(v)**params[3]+params[4]*np.exp(-params[5]*t)+params[6]*x*v+params[7]