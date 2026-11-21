def equation(t, P, params):
    p = params
    return (p[0] + p[1]*P + p[2]*P**2 + p[3]*P**3 + p[4]*P**4 + p[5]*P**5 + p[6]*P**6
            + p[7]/P + p[8]/P**2)
