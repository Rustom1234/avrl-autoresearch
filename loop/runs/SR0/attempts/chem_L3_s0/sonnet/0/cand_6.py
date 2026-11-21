def equation(t, A, params):
    return -params[0]+params[1]*A+params[2]*A*np.log(t)+params[3]*A**2+params[4]*np.log(t)*A**2