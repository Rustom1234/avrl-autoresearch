def equation(eps, T, params):
    a = params[0] + params[1]*T + params[2]*T**2
    b = params[3] + params[4]*T + params[5]*T**2
    return a - b*np.abs(eps)**(params[6])