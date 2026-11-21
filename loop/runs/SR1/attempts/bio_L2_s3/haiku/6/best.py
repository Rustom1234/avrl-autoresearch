def equation(t, P, params):
    s = np.sqrt(t)
    return (params[0] + params[1]*t + params[2]*s + params[3]*P + params[4]*P**2
            + params[5]*P**3 + params[6]*t*P + params[7]*s*P + params[8]*s*P**2)