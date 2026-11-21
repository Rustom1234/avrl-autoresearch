def equation(t, A, params):
    return (params[0]+A**params[1])*(1-np.exp(-params[3]*t))*(params[2]-params[4]*t+params[5]*t**2)