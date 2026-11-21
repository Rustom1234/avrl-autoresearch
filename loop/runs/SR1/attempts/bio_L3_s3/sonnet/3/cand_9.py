def equation(t,P,params):
    return (params[0]+params[1]*P**params[2])*np.exp(-(t-params[3])**2/params[4]**2)+params[5]*P