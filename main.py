import numpy as np
import matplotlib.pyplot as plt
import h5py
from powell import powell
import seaborn as sns

# Data loading functions
def get_extinction_coeficient(wavelength):
    """
    Retrieves extinction coefficient data for a given wavelength from HDF5 file.
    
    Parameters:
    -----------
    wavelength : int
        The wavelength (in nm) for which to retrieve extinction coefficient data
        
    Returns:
    --------
    ndarray
        Extinction coefficient values
    """
    with h5py.File('data.h5', 'r') as file:
        extinction = file[f'DataProducts/{wavelength}_ext'][:]
    return extinction

def get_backscattering_coeficient(wavelength):
    """
    Retrieves backscattering coefficient data for a given wavelength from HDF5 file.
    
    Parameters:
    -----------
    wavelength : int
        The wavelength (in nm) for which to retrieve backscattering coefficient data
        
    Returns:
    --------
    ndarray
        Backscattering coefficient values
    """
    with h5py.File('data.h5', 'r') as file:
        backscattering = file[f'DataProducts/{wavelength}_bsc'][:]
    return backscattering

def get_altitude():
    """
    Retrieves altitude data from HDF5 file.
    
    Returns:
    --------
    ndarray
        Altitude values
    """
    with h5py.File('data.h5', 'r') as file:
        altitude = file['DataProducts/Altitude'][:]
    return altitude

def compute_P(P_0, C, alpha, beta, delta_z):
    """
    Computes the lidar signal power using the lidar equation.
    
    Parameters:
    -----------
    P_0 : float
        Initial laser power
    C : float
        System constant
    alpha : ndarray
        Extinction coefficient profile
    beta : ndarray
        Backscattering coefficient profile
    delta_z : float
        Altitude step size
        
    Returns:
    --------
    ndarray
        Computed lidar signal power profile
    """
    n = alpha.size 
    P = np.zeros(n)
    # Compute cumulative integral of extinction coefficient
    integral_alpha = np.cumsum(alpha, axis=0) * delta_z
    
    # Apply lidar equation for each altitude level
    for i in range(1, n):
        P[i] = (P_0 * C * beta[i]) / (i * delta_z)**2 * np.exp(-2 * integral_alpha[i])
    return P

def get_valid_alpha_beta_indices(alpha, beta):
    """
    Filters out NaN values from extinction and backscattering coefficients
    and returns valid data ranges.
    
    Parameters:
    -----------
    alpha : ndarray
        Extinction coefficient profile
    beta : ndarray
        Backscattering coefficient profile
        
    Returns:
    --------
    tuple
        (filtered_alpha, filtered_beta, first_valid_index, last_valid_index)
    """
    # Identify valid (non-NaN) indices
    valid_indices = ~np.isnan(alpha) & ~np.isnan(beta)
    
    # Filter out NaNs
    alpha_valid = alpha[valid_indices]
    beta_valid = beta[valid_indices]
    
    # Get the first and last valid indices
    first_valid_index = np.where(valid_indices)[0][0]
    last_valid_index = np.where(valid_indices)[0][-1]
    
    return alpha_valid, beta_valid, first_valid_index, last_valid_index

def graph_coeficient(coeficient, wavelength):
    """
    Plots a coefficient profile for a given wavelength.
    
    Parameters:
    -----------
    coefficient : ndarray
        The coefficient values to plot
    wavelength : int
        The wavelength corresponding to the coefficient
    """
    plt.plot(coeficient)
    plt.title(f'{wavelength} Coeficient')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

def compute_J(P_signal, P_0, C, alpha, beta, delta_z):
    """
    Computes the cost function J using Simpson's rule integration.
    
    Parameters:
    -----------
    P_signal : ndarray
        Measured lidar signal
    P_0, C, alpha, beta, delta_z : 
        Parameters for lidar equation (see compute_P)
        
    Returns:
    --------
    float
        Computed cost function value
    """
    n = len(P_signal)
    P = np.zeros(n)
    integral_alpha = np.cumsum(alpha) * delta_z
    
    # Compute predicted signal
    for i in range(1, n):
        P[i] = (P_0 * C * beta[i]) / (i * delta_z)**2 * np.exp(-2 * integral_alpha[i])
    
    # Compute cost using Simpson's rule
    f = (P - P_signal)**2
    J_aprox = (delta_z / 3) * (f[0] + 4 * np.sum(f[1:n:2]) + 2 * np.sum(f[2:n-1:2]) + f[n-1])

    return J_aprox

def compute_penalty(alpha, beta):
    """
    Computes penalty term for negative coefficients.
    
    Parameters:
    -----------
    alpha : ndarray
        Extinction coefficient profile
    beta : ndarray
        Backscattering coefficient profile
        
    Returns:
    --------
    float
        Penalty value
    """
    phi = np.sum(np.maximum(0, -beta)**2 + np.maximum(0, -alpha)**2)
    return phi

def compute_J_penalized(J_aprox, penalty, rho):
    """
    Computes penalized cost function.
    
    Parameters:
    -----------
    J_aprox : float
        Approximate cost function value
    penalty : float
        Penalty term value
    rho : float
        Penalty weight factor
        
    Returns:
    --------
    float
        Penalized cost function value
    """
    J_penalized = J_aprox + rho * penalty
    return J_penalized

def optimize_J(z_final=10.0, P_signal=None, N=10, P_0=1.0, C=1.0, rho=1.0, alpha=None):
    """
    Optimizes the backscattering coefficient profile using Powell's method.
    
    Parameters:
    -----------
    z_final : float
        Maximum altitude
    P_signal : ndarray
        Measured lidar signal
    N : int
        Number of points in profile
    P_0 : float
        Initial laser power
    C : float
        System constant
    rho : float
        Penalty weight factor
    alpha : ndarray, optional
        Initial extinction coefficient profile
        
    Returns:
    --------
    tuple
        (optimized_beta, optimization_history)
    """
    if alpha is None:
        alpha = np.full(N, 0.01)

    def objective(x):
        beta = x
        J_aprox = compute_J(P_signal, P_0, C, alpha, beta, z_final / N)
        penalty = compute_penalty(alpha, beta)
        cost = compute_J_penalized(J_aprox, penalty, rho)
        return cost

    x0 = np.full(N, 0)
    result = powell(objective, x0)
    return result

if __name__ == "__main__":

        index = 100
        P_0 = 1  # Initial power
        C = 1    # System constant
        rho = 1.0  # Penalty weight
    
        # Load and preprocess coefficient data
        extinction_coef = get_extinction_coeficient(532)[index]
        backscattering_coef = get_backscattering_coeficient(532)[index]
        extinction_coef, backscattering_coef, first_valid_index, last_valid_index = get_valid_alpha_beta_indices(extinction_coef, backscattering_coef)
        N = len(extinction_coef)

        # Setup altitude grid and compute signal
        altitude = get_altitude().flatten()
        delta_z = (altitude[1] - altitude[0])/N
        z_final = delta_z * N
        P_signal = compute_P(P_0, C, alpha=extinction_coef, beta=backscattering_coef, delta_z=delta_z)

        # Optimize backscattering coefficient
        result, f_hist = optimize_J(z_final=z_final, P_signal=P_signal, N=N, P_0=P_0, C=C, rho=rho)
        beta_opt = result
        alpha_opt = np.zeros(N)
        P_predicted = compute_P(P_0, C, alpha_opt, beta_opt, delta_z)

        # Compute and store optimization results
        cost_1 = compute_J_penalized(compute_J(P_signal, P_0, C, alpha_opt, beta_opt, delta_z), 
                                   compute_penalty(alpha_opt, beta_opt), rho)
        
        # PLOTTING
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot extinction and backscattering coefficients
        axes[0].plot(extinction_coef, label='Extinction Coefficient', color=sns.color_palette("muted")[0])
        axes[0].plot(backscattering_coef, label='Backscattering Coefficient', color=sns.color_palette("muted")[1], linestyle='--')
        axes[0].set_title('532 nm Coefficients', fontsize=14)
        axes[0].set_xlabel('Index', fontsize=12)
        axes[0].set_ylabel('Coefficient Value', fontsize=12)
        axes[0].legend(loc="best")
        axes[0].grid(True)

        # Plot computed P signal
        axes[1].plot(P_signal, color=sns.color_palette("muted")[2])
        axes[1].set_title('P Signal', fontsize=14)
        axes[1].set_xlabel('Index', fontsize=12)
        axes[1].set_ylabel('Signal Value', fontsize=12)
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("original.png", format="png", dpi=300, bbox_inches="tight")
        plt.show()

        z_values = np.arange(N) * delta_z

        plt.figure(figsize=(18, 6))

        sns.set_style("whitegrid")
        sns.set_palette("muted")

        # First plot: P_signal vs P_predicted
        plt.subplot(1, 3, 1)
        plt.plot(z_values, P_signal, label='P_signal', color=sns.color_palette("muted")[0])
        plt.plot(z_values, P_predicted, label='P_predicted', linestyle='--', color=sns.color_palette("muted")[1])
        plt.xlabel('z', fontsize=12)
        plt.ylabel('P', fontsize=12)
        plt.legend()
        plt.title('P_signal vs P_predicted', fontsize=14)

        # Second plot: Backscattering Coefficient (Original vs Predicted)
        plt.subplot(1, 3, 2)
        plt.plot(z_values, backscattering_coef, label='Original', color=sns.color_palette("muted")[2])
        plt.plot(z_values, beta_opt, label='Predicted', linestyle='--', color=sns.color_palette("muted")[3])
        plt.xlabel('z', fontsize=12)
        plt.ylabel('Backscattering Coefficient', fontsize=12)
        plt.legend()
        plt.title('Backscattering Coefficient (Original vs Predicted)', fontsize=14)

        # Third plot: Cost function through iterations
        plt.subplot(1, 3, 3)
        plt.plot(range(len(f_hist)), f_hist, color=sns.color_palette("muted")[4])
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('J', fontsize=12)
        plt.title('Cost Function Through Iterations', fontsize=14)

        # Adjust layout for better display
        plt.tight_layout()
        plt.savefig("results.png", format="png", dpi=300, bbox_inches="tight")
        plt.show()