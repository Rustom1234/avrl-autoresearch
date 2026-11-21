def equation(t, A, params):
    return params[0]*A**2*np.sqrt(t) + params[1] + params[2]*np.log(t)+ params[3]/t + params[4]*A*t**params[5]