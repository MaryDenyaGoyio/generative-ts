import numpy as np

def RBF_kernel(T, tau):
    t = np.arange(T)
    return np.exp(-0.5 * ((t[:, None] - t[None, :])**2) / (tau**2))

def generate_data(**kwargs):

    if kwargs['name'] == 'example_1':
        prms = ["T", "std_Y", "v", "tau"]
        T, std_Y, v, tau = [kwargs[prm] for prm in prms]
        
        # e_t + θ^fixed + θ^GP_t
        noise_Y = np.random.normal(loc=0.0, scale=std_Y, size=T)

        theta_fixed = np.random.normal(loc=0.0, scale=v)

        theta_gp = np.random.multivariate_normal(mean=np.zeros(T), cov=RBF_kernel(T, tau))

        return (
                    (theta_fixed + theta_gp + noise_Y).reshape(T, 1),
                    {
                        r"\epsilon_{a,t}"         : noise_Y,
                        r"\theta^{\text{fixed}}_{a}" : theta_fixed,
                        r"\theta^{\text{GP}}_{a,t}"   : theta_gp
                    }
                )
    else:
        return None
