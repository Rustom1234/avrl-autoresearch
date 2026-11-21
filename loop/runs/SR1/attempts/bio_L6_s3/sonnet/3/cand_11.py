def equation(t,P,params):
    g=np.tanh(t/params[1])
    return -params[0]*g - params[2]*P*np.exp(-t/params[3]) + params[4]*g*P**params[5] + params[6]*g*P + params[7]*g