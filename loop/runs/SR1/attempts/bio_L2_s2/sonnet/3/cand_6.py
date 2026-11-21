def equation(t,P,params):
    x=np.log(P)
    return params[0]+params[1]*x+params[2]*t+params[3]*t*x+params[4]*x**2+params[5]*t*x**2