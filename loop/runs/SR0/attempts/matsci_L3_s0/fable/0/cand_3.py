def equation(eps, T, params):
    p=params
    return (p[0] + p[1]*eps**p[2])*(1-(T/p[3])**p[4]) + p[5]*eps**p[6]*T**p[7]