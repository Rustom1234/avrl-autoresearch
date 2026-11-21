def equation(t,P,params):
    return (params[0]*P**(-params[1])+params[2]+params[3]*P**params[4])*(1+params[5]*np.log(t))+params[6]*P