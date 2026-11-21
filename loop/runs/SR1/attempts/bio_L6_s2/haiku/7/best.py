def equation(t, P, params):
    lP=np.log(P)
    return (params[0]*P + params[1]*P**2 + params[2]*P*lP + params[3]*P*lP*t + params[4]*P*t
            + params[5]*P**2*t + params[6]*t + params[7]*P**3 + params[8]*P*t**2 + params[9]*P*lP*t**2)