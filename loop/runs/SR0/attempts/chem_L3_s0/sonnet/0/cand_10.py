def equation(t, A, params):
    return -params[0]*np.exp(-params[1]*A)+params[2]*A*np.exp(-params[3]*A)*t**params[4]+params[5]*A*t**params[6]