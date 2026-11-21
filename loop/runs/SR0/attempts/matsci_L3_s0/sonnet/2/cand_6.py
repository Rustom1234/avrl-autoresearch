def equation(eps,T,params):
    return (params[0]+params[1]*T)*eps + params[2]*np.sqrt(eps)*np.exp(-params[3]*(T-params[4])**2) + params[5]*eps**2