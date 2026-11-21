def equation(t,P,params):
    return params[0]*(1-P/params[1])*(P-params[2])/(P+params[3])*P**params[4]