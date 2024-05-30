import numpy as np

def simulated_annealing(f, x, d=0.5, a=0.99, t0 = 10, K=100):
    n = len(x)
    results = {"x_optimal": x, 
               "f_optimal": f(x),
               "x_history": np.full((K, n), None),
               "f_history": np.full(K, None),
               "acceptance_probability_k": np.full(K, None),
               "temperature_k": np.full(K, None)}
    results["x_history"][0, :] = x
    results["f_history"][0] = f(x)
    temperature_k = t0
    for k in range(2, K):
        xc = x + np.random.uniform(-d, d, size=n)
        delta_t = -(f(xc)-f(x))
        acceptance_probability_k = min(1, np.exp(delta_t/temperature_k))
        results["acceptance_probability_k"][k] = acceptance_probability_k

        if np.random.uniform(0,1) < acceptance_probability_k:
            x = xc
            if f(x) < results["f_optimal"]:
                results["x_optimal"] = x
                results["f_optimal"] = f(x)
        results["x_history"][k, :] = x
        results["f_history"][k] = f(x)
        temperature_k = a*temperature_k
    return results

def f(x):
    return np.sum((x**2 - 4*x + 4))

# Initial solution (2-dimensional example)
x = np.array([0])
r = simulated_annealing(f=f,x=x)
print(r["x_optimal"]) 