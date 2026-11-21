def equation(t, A, params):
    return -params[0]+params[1]*A**params[2]/(params[3]+A**params[2])*np.log(t+params[4])+params[5]*A