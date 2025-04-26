##### Import necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm as tqdm         # tqdm is a nice library to visualize ongoing loops



#--------------------------------
# Parameters and primitives
#--------------------------------

# Parameters governing preferences and technology
bbeta=0.99        # Discount factor    
ppsi = 2.4        # Disutility of labor
aalpha = 0.256    # Capital share in production function
nnu = 0.64        # Labor share in production function
ddelta = 0.069    # Capital depreciation rate
eeta = 0.5        # Capital adjustment cost

rrho_z = 0.95     # Autocorrelation of idiosyncratic productivity shock
ssigma_z = 0.05   # Standard deviation of idiosyncratic productivity shock

# Parameters governing the ergodic distribution of shocks
ssigma_e_z = ssigma_z / math.sqrt(1.0 - rrho_z**2) # Standard deviation of ergodic productivity distribution
k_min = 1e-3  # Minimum value of capital 
k_max = 20.0  # Maximum value of capital



#--------------------------------
# Set up neural networks
#--------------------------------

# Construction of neural network
layers = [
    tf.keras.layers.Dense(128, activation='relu', input_dim=2, bias_initializer='he_uniform'),
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
            new_bias   = layer.bias_initializer(shape=layer.bias.shape)
            layer.set_weights([new_kernel, new_bias])


#--------------------------------
# Policy, residuals, and objective
#--------------------------------

# Define firm policy function k'(k,z)
def k_policy(k,z):

    # Normalize the two states
    norm_k = (k - k_min) / (k_max - k_min) * 2.0 - 1.0 # Normalize k so it lies within [-1,1]
    norm_z = z / ssigma_e_z / 3.0     # Normalize z so they are typically between -1 and 1

    # Stack into [n by 2] matrix
    s = tf.stack([norm_k, norm_z], axis=1)
    x = policy_net(s)

    # Capital policy is nested within [k_min, k_max]
    # kp = k_min + (k_max - k_min) * tf.sigmoid(x[:, 0])
    kp = 0.001 * k_min + (1.3 * k_max - 0.001 * k_min) * tf.sigmoid(x[:, 0])  # Capital policy might be off-the grid, but at the same time the upper bound of it shouldn't be too high, 
                                                              # otherwise the ADAM optimizer will be unstable.

    return kp


# Define the optimal labor given states (k,z) and wage rate w
def optimal_labor(k,z,w):

    # Compute the optimal labor choice
    n = (nnu * tf.exp(z) * k**aalpha / w)**(1.0 / (1.0 - nnu))

    return n


# Define the FOC residual
def residuals(e_z, k, z, w):
    
    # Current period's labor and investment decisions
    kp = k_policy(k,z)          # Optimal investment k'(k,z)

    # Tomorrow's states and optimal decisions
    z_next = z * rrho_z + e_z   # Tomorrow's productivity
    np_next = optimal_labor(kp,z_next,w) # Tomorrow's optimal labor n'(k',z')
    kp_next = k_policy(kp, z_next) # Tomorrow's investment decision k''(k',z')

    # Intertemporal Euler FOC residual
    R = (1.0 + eeta * (kp - k)) - bbeta*(tf.exp(z_next) * aalpha * (kp_next**(aalpha-1.0)) * (np_next**(nnu)) + 1.0 - ddelta + eeta * (kp_next - kp))

    return R


# Objective function to be minimized
def objective_function(n, w):
    
    # randomly drawing current states
    z = tf.random.normal(shape=(n,), stddev=ssigma_e_z)
    k = tf.random.uniform(shape=(n,), minval=k_min, maxval=k_max)

    # Randomly draw 1st and 2nd realizations of productivity shocks
    e1_z = tf.random.normal(shape=(n,), stddev=ssigma_z)
    e2_z = tf.random.normal(shape=(n,), stddev=ssigma_z)

    # Compute the residuals
    R1 = residuals(e1_z, k, z, w)
    R2 = residuals(e2_z, k, z, w)

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
llambda = 1e-3          # Learning rate of ADAM

# Initialization
iter = 0            # Number of iteration
restart = 0         # Number of restart
p_lb = 0.5          # Price lower bound
p_ub = 10.0         # Price upper bound
maxIter = 1
maxRestart = 5      # Maximum number of restarting neural network training due to explosive solution
# maxIter = 30      # Maximum number of bisection
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


# Bisection 
t0 = time.time()
while (iter < maxIter) and (restart < maxRestart):

    results = []
    
    # Compute the midpoint price
    p_trial = 0.5 * (p_lb + p_ub)

    # Compute corresponding wage rate
    w_trial = ppsi / p_trial 

    # Reinitialize the neural network weights and biases
    reinitialize_model()
    ttheta = policy_net.trainable_variables
    optimizer = tf.keras.optimizers.Adam(learning_rate=llambda, amsgrad=True)
    
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

            # Productivity shock:
            z_shock = eps_z[tt]

            # Compute optimal labor choice
            n_simul = optimal_labor(k_simul,z_simul,w_trial)

            # Compute firm's investment decision
            kp_simul = k_policy(k_simul,z_simul)
            kp_simul = tf.clip_by_value(kp_simul, k_min, k_max)

            # Compute aggregates
            Y_agg[tt] = float(tf.reduce_mean(tf.exp(z_simul) * (k_simul**aalpha) * (n_simul**nnu)))
            z_agg[tt] = float(tf.reduce_mean(z_simul))
            K_agg[tt] = float(tf.reduce_mean(k_simul))
            I_agg[tt] = float(tf.reduce_mean(kp_simul - (1.0 - ddelta) * k_simul))
            AC_agg[tt] = float(tf.reduce_mean(0.5 * eeta * (kp_simul - k_simul)**2))
            C_agg[tt] = Y_agg[tt] - I_agg[tt] - AC_agg[tt]

            # Update capital state
            k_simul = kp_simul

            # Update productivity state
            z_simul = tf.clip_by_value(rrho_z*z_simul + z_shock, z_min, z_max)

        # Compute the aggregate consumption 
        C_eq = C_agg[T_burn:].mean()

        # Compute the implied equilibrium price
        p_eq = 1.0 / C_eq

        if (C_eq > 0) and (C_eq < k_max):
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


# 2) Plot policy function kp = k_policy(k,z) for various z ---------------------
# Choose a grid of k
k_grid = np.linspace(k_min, k_max, 200, dtype=np.float32)

# Choose a few z’s to visualize
zbar = z_agg[T_burn:].mean()
z_vals = [-3 * ssigma_z + zbar, zbar, 3 * ssigma_z + zbar]
plt.figure(figsize=(6,4))
for z0 in z_vals:
    # Broadcast k_grid and z0 into tensors
    k_tf = tf.constant(k_grid)
    z_tf = tf.constant(np.ones_like(k_grid) * z0, dtype=tf.float32)
    kp_tf = k_policy(k_tf, z_tf)     # calls your trained network internally
    kp_np = kp_tf.numpy()
    plt.plot(k_grid, kp_np, label=f"z = {z0:.2f}")

# 45° line for reference
plt.plot(k_grid, k_grid, 'k--', label='45°')
plt.xlabel(r'$k_t$')
plt.ylabel(r"$k_{t+1} = \mathrm{policy}(k_t,z_t)$")
plt.title("Capital Policy Function")
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()



# 3) Plot policy function kp = k_policy(k,z) for various k ---------------------
# Set up a grid in z‑space
z_grid = np.linspace(z_min, z_max, 200, dtype=np.float32)

# Pick a few capital levels to visualize
kbar = K_agg[T_burn:].mean()
k_levels = [0.5 * kbar, kbar, 1.25*kbar]  # adjust to your economically relevant k’s
plt.figure(figsize=(6,4))
for k0 in k_levels:
    # make TensorFlow constants of shape [200]
    k_tf = tf.constant(np.ones_like(z_grid) * k0, dtype=tf.float32)
    z_tf = tf.constant(z_grid, dtype=tf.float32)
    # compute the capital policy k'(k,z)
    kp_tf = k_policy(k_tf, z_tf)
    # convert to NumPy and plot
    plt.plot(z_grid, kp_tf.numpy(), label=f"$k_t = {k0}$")

# reference line k'=k
plt.plot(z_grid, z_grid*0 + np.array(k_levels)[0], 'k--', alpha=0.7,
         label=f"$k'=k$")

plt.xlabel(r"$z_t$")
plt.ylabel(r"$k_{t+1} = \pi(k_t,z_t)$")
plt.title("Policy $k'(k,z)$ at different $k$")
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Plot consumption
plt.figure(figsize=(6,4))
plt.plot(C_agg, label='aggregate consumption')
plt.show()
