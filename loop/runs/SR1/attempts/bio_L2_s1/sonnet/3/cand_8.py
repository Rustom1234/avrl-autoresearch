def equation(t,P,params):
    return (params[0]+params[1]*t**params[2])*np.exp(-(P/params[3])**params[4])