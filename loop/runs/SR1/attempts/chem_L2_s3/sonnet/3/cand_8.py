def equation(t,A,params):
    g=1-np.exp(-params[3]*t)
    return -(params[0]+params[1]*np.exp(-params[2]*t)+ params[4]*g*A**2 + params[5]*A + params[6]*A*np.exp(-params[7]*t)+params[8]*g)