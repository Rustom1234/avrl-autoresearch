def equation(t, P, params):
    return (params[0] + params[1]*t**2 + params[6]*t)*P + (params[2] + params[3]*t**2)*P*np.log(P) + params[4] + params[5]*P**2 + params[7]*t**2*P**2