def equation(t,P,params):
    return P*(params[0]-params[1]*P)-(params[2]+params[3]*t+params[4]*t**2+params[5]*t**3)*P**params[6]