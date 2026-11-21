def equation(t,A,params):
    return -(params[0]/t**params[1]+params[2]*t**params[3]+params[4]*A**params[5]*t**params[6]+params[7]*A)