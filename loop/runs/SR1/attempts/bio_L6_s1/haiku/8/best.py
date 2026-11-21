def equation(t, P, params):
    L = np.log(P)
    return (params[0] + params[1]*P + params[2]*P**2 + params[3]*t + params[4]*t*P
            + params[5]*t**2 + params[6]*t*L + params[7]*P*L + params[8]*t*P*L + params[9]*L)