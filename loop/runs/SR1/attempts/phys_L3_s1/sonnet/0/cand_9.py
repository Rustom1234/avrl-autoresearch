def equation(t,x,v,params):
    return (params[0]+params[1]*v+params[8]*v**2)*np.cos(params[2]*t+params[3])+params[4]-params[5]*v-params[6]*v**2-params[7]*x