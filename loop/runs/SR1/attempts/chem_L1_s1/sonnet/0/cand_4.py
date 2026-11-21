def equation(t, A, params):
    return (params[0]/A + params[1] + params[2]*A + params[3]*A**2)+params[4]*np.sin(params[5]*t+params[6])