def equation(eps, T, params):
    p=params
    return (p[0]+p[1]*np.abs(eps)**p[2])*(1-(np.abs(T-p[3])/p[4])**p[5])