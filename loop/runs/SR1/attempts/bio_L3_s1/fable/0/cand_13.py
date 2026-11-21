def equation(t, P, params):
    return (params[0]*t + params[7])*np.sin(params[1]*P) + params[2]/P + params[3]*P + params[4]*P**2/(params[5]**2 + P**2) + params[6]*P**2 + params[8]