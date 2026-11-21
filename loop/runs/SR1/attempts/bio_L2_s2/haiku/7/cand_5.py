def equation(t,P,params):
    a,b,w,ph,c,d,ps=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return P*(a+b*np.sin(w*t+ph)) - P**2*(c+d*np.sin(w*t+ps))