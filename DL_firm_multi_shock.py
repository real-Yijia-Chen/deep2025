##### Import necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time
from tqdm import tqdm as tqdm         # tqdm is a nice library to visualize ongoing loops



#--------------------------------
# Parameters and primitives
#--------------------------------

# Parameters governing preferences and technology
bbeta=0.99          # Discount factor    
ppsi = 2.4          # Disutility of labor
aalpha = 0.256      # Capital share in production function
nnu = 0.64          # Labor share in production function
ddelta = 0.069      # Capital depreciation rate
eeta = 0.5          # Capital adjustment cost

rrho_z = 0.95       # Autocorrelation of persistent productivity shock
ssigma_z = 0.05     # Standard deviation of persistent productivity shock
ssigma_ups = 0.03   # Standard deviation of transitory productivity shock 
ssigma_omega = 0.04 # Standard deviation of capital quality shock
omega_max = 0       # Upper bound of capital quality shock
omega_min = -4 * ssigma_omega  # Lower bound of capital quality shock

# Parameters governing the ergodic distribution of shocks
ssigma_e_z = ssigma_z / math.sqrt(1.0 - rrho_z**2) # Standard deviation of ergodic productivity distribution
k_min = 0           # Minimum value of capital 
k_max = 20.0        # Maximum value of capital



#--------------------------------
# Set up neural networks
#--------------------------------

# Construction of neural network
layers = [
    tf.keras.layers.Dense(128, activation='relu', input_dim=4, bias_initializer='he_uniform'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
]
policy_net = tf.keras.Sequential(layers)

# helper to re‐initialize all Dense layers’ kernels & biases
def reinitialize_model():
    he = tf.keras.initializers.HeUniform()
    for layer in policy_net.layers:
        # only reinit Dense layers that actually have kernel & bias
        if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
            # sample new weights with He‐uniform
            new_kernel = he(shape=layer.kernel.shape)
            # sample new biases with whatever initializer you originally set
            new_bias   = layer.bias_initializer(shape=layer.bias.shape, dtype=layer.bias.dtype)
            layer.set_weights([new_kernel, new_bias])



#--------------------------------
# Policy, residuals, and objective
#--------------------------------

# Define firm policy function k'(k,z)
def k_policy(k,z,ups,omega):

    # Normalize the states
    norm_k = (k - k_min) / (k_max - k_min) * 2.0 - 1.0 # Normalize k so it lies within [-1,1]
    norm_z = z / ssigma_e_z / 3.0     # Normalize z so they are typically between -1 and 1
    norm_ups = ups / ssigma_ups / 3.0   # Normalize ups so they are typically between -1 and 1
    norm_omega = (omega - omega_min) / (omega_max - omega_min) * 2.0 - 1.0    # Normalize omega

    # Stack into [n by 4] matrix
    s = tf.stack([norm_k, norm_z, norm_ups, norm_omega], axis=1)
    x = policy_net(s)

    # Capital policy is nested within [k_min, k_max]
    kp = k_min + (1.3 * k_max - k_min) * tf.sigmoid(x[:, 0])

    return kp


# Define the optimal labor given states (k,z) and wage rate w
def optimal_labor(k,z,ups,omega,w):

    # Compute the optimal labor choice
    n = (nnu * tf.exp(z) * tf.exp(ups) * (tf.exp(omega) * k)**aalpha / w)**(1.0 / (1.0 - nnu))

    return n


# Define the FOC residual
def residuals(e_z, ups_next, omega_next, k, z, ups, omega, w):
    
    # Current period's labor and investment decisions
    kp = k_policy(k,z,ups,omega)          # Optimal investment k'(k,z)

    # Tomorrow's states and optimal decisions
    z_next = z * rrho_z + e_z   # Tomorrow's persistent productivity
    np_next = optimal_labor(kp,z_next,ups_next,omega_next,w) # Tomorrow's optimal labor n'
    kp_next = k_policy(kp,z_next,ups_next,omega_next) # Tomorrow's investment decision k''

    # Intertemporal Euler residual
    R = (1.0 + eeta * (kp - tf.exp(omega) * k)) - bbeta*(tf.exp(omega_next) * tf.exp(z_next) * tf.exp(ups_next) * aalpha * 
        ((tf.exp(omega_next) * kp_next)**(aalpha-1.0)) * (np_next**(nnu)) + (1.0 - ddelta) * tf.exp(omega_next) 
        + eeta * tf.exp(omega_next) * (kp_next - tf.exp(omega_next) * kp))

    return R


# Objective function to be minimized
def objective_function(n, w):
    
    # randomly drawing current states
    z = tf.random.normal(shape=(n,), stddev=ssigma_e_z)
    k = tf.random.uniform(shape=(n,), minval=k_min, maxval=k_max)
    ups = tf.random.normal(shape=(n,), stddev=ssigma_ups)
    omega = tf.random.normal(shape=(n,), stddev=ssigma_omega)
    omega = tf.clip_by_value(omega, clip_value_min=omega_min, clip_value_max=omega_max)

    # Randomly draw 1st and 2nd realizations of shocks
    e1_z = tf.random.normal(shape=(n,), stddev=ssigma_z)
    e2_z = tf.random.normal(shape=(n,), stddev=ssigma_z)
    ups_1 = tf.random.normal(shape=(n,), stddev=ssigma_ups)
    ups_2 = tf.random.normal(shape=(n,), stddev=ssigma_ups)
    omega_1 = tf.random.normal(shape=(n,), stddev=ssigma_omega)
    omega_1 = tf.clip_by_value(omega_1, clip_value_min=omega_min, clip_value_max=omega_max)
    omega_2 = tf.random.normal(shape=(n,), stddev=ssigma_omega)
    omega_2 = tf.clip_by_value(omega_2, clip_value_min=omega_min, clip_value_max=omega_max)

    # Compute the residuals
    R1 = residuals(e1_z, ups_1, omega_1, k, z, ups, omega, w)
    R2 = residuals(e2_z, ups_2, omega_2, k, z, ups, omega, w)

    # All-in-one (AiO) expectation
    R_squared = R1 * R2 

    # Compute the average
    return tf.reduce_mean(R_squared)



#--------------------------------
# Computing equilibrium
#--------------------------------

# Parameters governing the training, simulation, and bisection
z_min = -3.0 * ssigma_e_z    # Minimum productivity
z_max = 3.0 * ssigma_e_z     # Maximum productivity
N_sample = 512          # Number of samples in Adam optimizer
N_training = 50000      # Number of training steps
N_errs_check = N_training-1000  # Starting number of error checking (to rule out explosive solutions) 
N_firm = 500000         # Number of firms in the simulation
T_simul = 200           # Simulation period
T_burn = 100            # Period burned
tol_err = 0.1           # Tolerance of Euler error

# Initialization
iter = 0            # Number of iteration
restart = 0         # Number of restart
p_lb = 0.5          # Price lower bound
p_ub = 10.0         # Price upper bound
maxIter = 12        # Maximum number of bisection
maxRestart = 10     # Maximum number of restarting neural network training due to explosive solution
# maxIter = 30      
tol = 1e-2          # Tolerance for consumption market clearing
p_in = np.zeros(maxIter) # Store the price guess (for debugging)
p_out = np.zeros(maxIter) # Store the price implied (for debugging)

# Fix a global seed so everything is reproducible
tf.random.set_seed(1234)
np.random.seed(1234)

# Pre-draw initial cross-section of firm states
k0 = tf.ones(N_firm, dtype=tf.float32) * 0.1
z0 = tf.random.normal(shape=(N_firm,), stddev=ssigma_e_z)

# Pre-draw all shocks for all periods
eps_z = tf.random.normal(shape=(T_simul, N_firm),mean=0.0,stddev=ssigma_z)
eps_ups = tf.random.normal(shape=(T_simul, N_firm),mean=0.0,stddev=ssigma_ups)
eps_omega = tf.random.normal(shape=(T_simul, N_firm),mean=0.0,stddev=ssigma_omega)
eps_omega = tf.clip_by_value(eps_omega, clip_value_min=omega_min, clip_value_max=omega_max)


# Bisection 
t0 = time.time()
while (iter < maxIter) and (restart < maxRestart):

    results = []
    
    # Compute the midpoint price
    p_trial = 0.5 * (p_lb + p_ub)

    # Compute corresponding wage rate
    w_trial = ppsi / p_trial 


    #--------------------------------
    #    Set up ADAM optimizer
    #--------------------------------

    # Reinitialize the neural network weights and biases
    reinitialize_model()
    ttheta = policy_net.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    
    @tf.function
    def training_step(m,w):

        with tf.GradientTape() as tape:
            xx = objective_function(m,w)

        grads = tape.gradient(xx, ttheta)
        optimizer.apply_gradients(zip(grads,ttheta))

        return xx

    def train_me(N_training,N_sample,w):
        vals = []
        for k in tqdm(range(N_training)):
            val = training_step(N_sample,w)
            vals.append(val.numpy())

        return vals
    
    # Train the neural network at given price
    results = train_me(N_training, N_sample, w_trial)

    # Check the error
    errs = np.array(results)
    max_rmse = np.sqrt(max(errs[N_errs_check:]))

    if max_rmse <= tol_err: # Update the price only if Euler error is below the threshold

        #--------------------------------
        # Simulate the firm distribution
        #--------------------------------

        # Vectors that store the aggregates
        Y_agg = np.zeros(T_simul)
        z_agg = np.zeros(T_simul)
        K_agg = np.zeros(T_simul)
        I_agg = np.zeros(T_simul)
        AC_agg = np.zeros(T_simul)
        C_agg = np.zeros(T_simul)

        # Impose initial states
        k_simul = k0
        z_simul = z0

        # Simulation
        for tt in range(T_simul):

            # Shocks:
            z_shock = eps_z[tt]
            ups_shock = eps_ups[tt]
            omega_shock = eps_omega[tt]

            # Compute optimal labor choice
            n_simul = optimal_labor(k_simul,z_simul,ups_shock,omega_shock,w_trial)

            # Compute firm's investment decision
            kp_simul = k_policy(k_simul,z_simul,ups_shock,omega_shock)
            kp_simul = tf.clip_by_value(kp_simul, clip_value_min=k_min, clip_value_max=k_max)

            # Compute aggregates
            Y_agg[tt] = float(tf.reduce_mean(tf.exp(z_simul) * tf.exp(ups_shock) * ((tf.exp(omega_shock) * k_simul)**aalpha) * (n_simul**nnu)))
            z_agg[tt] = float(tf.reduce_mean(z_simul))
            K_agg[tt] = float(tf.reduce_mean(tf.exp(omega_shock) * k_simul))
            I_agg[tt] = float(tf.reduce_mean(kp_simul - (1.0 - ddelta) * tf.exp(omega_shock) * k_simul))
            AC_agg[tt] = float(tf.reduce_mean(0.5 * eeta * (kp_simul - tf.exp(omega_shock) * k_simul)**2))
            C_agg[tt] = Y_agg[tt] - I_agg[tt] - AC_agg[tt]

            # Update capital state
            k_simul = kp_simul

            # Update productivity state
            z_simul = tf.clip_by_value(rrho_z*z_simul + z_shock, clip_value_min=z_min, clip_value_max=z_max)

        # Compute the aggregate consumption 
        C_eq = C_agg[T_burn:].mean()

        # Compute the implied equilibrium price
        p_eq = 1.0 / C_eq

        # Store the two prices
        p_in[iter] = p_trial
        p_out[iter] = p_eq

        # Check convergence
        C_trial = 1.0 / p_trial
        if abs(C_trial - C_eq) < tol: 
            break

        # Update bounds
        if p_trial < p_eq:
            # Guess was too low → move lower bound up
            p_lb = p_trial
        else:
            # Guess was too high → move upper bound down 
            p_ub = p_trial

        # Update number of iteration
        iter = iter + 1
    else: 
        restart = restart + 1



# # 1) Plot training error -----------------------------------------------------
# `results` should be a Python list or 1D array of the objective (R_squared) at each epoch.
errs = np.array(results)
plt.figure(figsize=(6,4))
plt.plot(np.sqrt(errs), label='training RMSE')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch (log scale)')
plt.ylabel('Training error (log scale)')
plt.title('Training Error')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# 2) Plot kp = k_policy(k,z) for various z, holding ups=0, omega=0
plt.close()
k_grid = np.linspace(k_min, k_max, 200, dtype=np.float32)
zbar   = z_agg[T_burn:].mean()
z_vals = [zbar - 3*ssigma_z, zbar, zbar + 3*ssigma_z]

plt.figure(figsize=(6,4))
for z0 in z_vals:
    k_tf     = tf.constant(k_grid)
    z_tf     = tf.constant(np.ones_like(k_grid)*z0, dtype=tf.float32)
    ups_tf   = tf.zeros_like(k_tf)      # hold ups=0
    omega_tf = tf.zeros_like(k_tf)      # hold omega=0

    kp_tf = k_policy(k_tf, z_tf, ups_tf, omega_tf)
    plt.plot(k_grid, kp_tf.numpy(), label=f"$z={z0:.2f}$")

# 45° reference
# plt.plot(k_grid, k_grid, 'k--', label='45°')
plt.xlabel(r"$k_t$")
plt.ylabel(r"$k_{t+1} = \mathrm{policy}(k_t,z_t,\bar\upsilon,\bar\omega)$")
plt.title("Capital Policy at various $z$ (ups=0, omega=0)")
plt.legend(); plt.grid(ls='--', alpha=0.5); plt.tight_layout()
plt.show()



# 3) Plot kp = k_policy(k,z) for various k, holding ups=0, omega=0
plt.close()
z_grid = np.linspace(z_min, z_max, 200, dtype=np.float32)
kbar   = K_agg[T_burn:].mean()
k_levels = [0.5*kbar, kbar, 1.25*kbar]

plt.figure(figsize=(6,4))
for k0 in k_levels:
    k_tf     = tf.constant(np.ones_like(z_grid)*k0, dtype=tf.float32)
    z_tf     = tf.constant(z_grid, dtype=tf.float32)
    ups_tf   = tf.zeros_like(z_tf)      # hold ups=0
    omega_tf = tf.zeros_like(z_tf)      # hold omega=0

    kp_tf = k_policy(k_tf, z_tf, ups_tf, omega_tf)
    plt.plot(z_grid, kp_tf.numpy(), label=f"$k={k0:.2f}$")

# reference: k'=k
# plt.plot(z_grid, z_grid*0 + k_levels[0], 'k--', alpha=0.7, label="45°")
plt.xlabel(r"$z_t$")
plt.ylabel(r"$k_{t+1} = \mathrm{policy}(k_t,z_t,\bar\upsilon,\bar\omega)$")
plt.title("Capital Policy at various $k$ (ups=0, omega=0)")
plt.legend(); plt.grid(ls='--', alpha=0.5); plt.tight_layout()
plt.show()


# Plot consumption
plt.close()
plt.figure(figsize=(6,4))
plt.plot(C_agg, label='aggregate consumption')
plt.show()
