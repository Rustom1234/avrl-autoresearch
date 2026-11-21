def equation(t, A, params):
    return params[0] + params[1]*np.log(t) - params[2]*A**2*np.sqrt(t) - params[3]/t - params[4]*A*np.log(t)- params[5]*A**2/t