def equation(t,P,params):
    return (params[0]+params[1]*P**params[2])*(1+params[3]*t*np.exp(-t/params[4]))