def equation(t, A, params):
    T=1+0*t
    s=np.sqrt(A)
    return (params[0]+params[1]*t+params[2]*t*t)*s+(params[3]+params[4]*t+params[5]*t*t)*A+(params[6]+params[7]*t+params[8]*t*t)*np.log(A)+params[9]