def equation(t, P, params):
    return (params[0]*P + params[1]*P**2 + params[2]*P**3 + params[3]*P*t + params[4]*P*t**2
            + params[5]*P**2*t + params[6]*P*np.log(t) + params[7]*P*np.sqrt(t) + params[8]*t + params[9]*P**2*np.log(t))