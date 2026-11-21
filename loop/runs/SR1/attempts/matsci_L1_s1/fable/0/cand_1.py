def equation(eps, T, params):
    return (params[0]+params[1]*np.abs(eps)**params[2])*(1-(np.abs((T-params[3])/params[4]))**params[5])