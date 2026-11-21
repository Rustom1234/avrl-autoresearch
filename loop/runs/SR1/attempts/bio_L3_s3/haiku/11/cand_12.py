def equation(t,P,params):
    a,b,c,d,e,f,g=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    u=t/(1+t)
    return a + b*P + c*P**2 + d*u + e*P*u + f*t + g*P*t
