def equation(t, A, params):
    return params[0]*A**params[1] + (params[2]+params[3]*t)*A**params[4] + params[5]*t + params[6]*np.log(A)+params[7]*t*np.log(A)