def equation(t, A, params):
    return params[0]+params[1]*np.log(A)+(params[2]*t**2+params[3])/A