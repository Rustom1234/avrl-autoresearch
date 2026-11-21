def equation(t, P, params):
    a,b,c,d,e,f,g,h,k=params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return P*(a + b*P + c*P**2 + d*t + e*t**2 + f*t*P + g*t**3 + h*P/(1+P) + k*t*P**2)
