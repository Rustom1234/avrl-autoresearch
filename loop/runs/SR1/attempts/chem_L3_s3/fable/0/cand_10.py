def equation(t, A, params):
    return 7.3*params[0]/t + 2.59*params[1]*np.sin(1.46*params[2]*np.sqrt(t)+0.02*params[3]) - 0.197*params[4]*np.log(1+0.538*params[5]*t)*A**2