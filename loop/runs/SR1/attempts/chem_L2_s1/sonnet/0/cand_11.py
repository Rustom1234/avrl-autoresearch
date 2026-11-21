def equation(t,A,params):
    return -(A**params[0]*(params[1]-params[2]*t**params[3]) + params[4]*t**params[5])