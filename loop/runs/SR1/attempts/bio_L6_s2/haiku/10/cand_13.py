def equation(t, P, params):
    return (params[0]*P + params[1]*P**2 + params[2]*P**3 + params[3]*P*np.log(P)
            + params[4]*t*P + params[5]*t*P**2 + params[6]*t**2*P + params[7]*t**2*P**2)
