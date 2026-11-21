def equation(t, P, params):
    num = params[0] + params[1]*P + params[2]*P**2 + params[3]*P**3
    den = 1 + params[4]*P + params[5]*P**2
    return num/den
