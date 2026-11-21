def equation(t,P,params):
    return -params[0]*np.tanh(t/params[1]) - params[2]*P + params[3]*P**2*t**params[4]+params[5]*P*np.tanh(t/params[1])