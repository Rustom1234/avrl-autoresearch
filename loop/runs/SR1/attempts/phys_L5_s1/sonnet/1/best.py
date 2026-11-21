def equation(t,x,v,params):
    L=np.log(t)
    return params[0]+params[1]*v+params[2]*v**2+params[3]*v**3+params[4]*L+params[5]*L**2+params[6]*x+params[7]*L**3