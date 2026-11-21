def equation(t, P, params):
    cols=[t**i*P**j for i in range(5) for j in range(5) if i+j<=4]
    return sum(params[k]*cols[k] for k in range(len(cols)))