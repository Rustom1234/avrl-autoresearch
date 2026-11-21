def equation(t, A, params):
    return (params[0]+params[1]*np.log(A)+params[7]*A)*np.exp(-params[2]*t) + params[3]*A**params[4]*np.exp(-params[5]*t)+params[6]