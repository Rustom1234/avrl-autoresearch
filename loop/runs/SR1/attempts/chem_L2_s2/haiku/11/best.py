def equation(t, A, params):
    return (params[0] + params[1]*A + params[2]*A*A + params[3]/A + params[4]*t/A + params[5]*t
            + params[6]*A*t + params[7]*np.log(A) + params[8]*t*np.log(A) + params[9]*np.sqrt(A))