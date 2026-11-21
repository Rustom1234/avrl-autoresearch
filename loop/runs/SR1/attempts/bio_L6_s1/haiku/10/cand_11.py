def equation(t, P, params):
    return (params[0] + params[1]*t + params[2]*t**2)*P*np.log(params[3]/P) + params[4] + params[5]*P + params[6]*P**2