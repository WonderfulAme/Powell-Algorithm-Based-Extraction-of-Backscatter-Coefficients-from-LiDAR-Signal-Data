import numpy as np

def line_search(f, x, direction, alpha_start=1.0, max_iter=1000, tol=1e-10):
    """
    Performs an enhanced line search to find the optimal step size (alpha) in a given direction.
    Uses a combination of bracketing and golden section search for efficient optimization.
    
    Parameters:
    -----------
    f : callable
        Objective function to minimize
    x : ndarray
        Current point in the search space
    direction : ndarray
        Search direction
    alpha_start : float, optional (default=1.0)
        Initial step size
    max_iter : int, optional (default=1000)
        Maximum number of iterations
    tol : float, optional (default=1e-10)
        Tolerance for convergence
        
    Returns:
    --------
    float
        Optimal step size (alpha) that minimizes f(x + alpha * direction)
    """
    alpha = alpha_start
    f_current = f(x)
    
    # Initialize bracketing bounds
    alpha_low = 0.0  # Lower bound of bracket
    alpha_high = alpha_start  # Upper bound of bracket
    
    # Golden section constants
    golden_ratio = (1 + np.sqrt(5)) / 2  # ≈ 1.618034 (phi)
    golden_section = 2 - golden_ratio  # ≈ 0.381966
    
    # Phase 1: Bracket the minimum by expanding the interval
    # We keep doubling alpha_high until we find a bracket that contains a minimum
    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)
        
        if f_new < f_current:
            # If we found a better point, expand the bracket
            alpha_high *= 2
            x_test = x + alpha_high * direction
            f_test = f(x_test)
            if f_test > f_new:
                # We've bracketed a minimum when the function starts increasing
                break
            alpha = alpha_high
            f_current = f_test
        else:
            break
    
    # Phase 2: Golden section search
    # Initialize the two internal points for golden section search
    alpha1 = alpha_low + golden_section * (alpha_high - alpha_low)
    alpha2 = alpha_high - golden_section * (alpha_high - alpha_low)
    f1 = f(x + alpha1 * direction)
    f2 = f(x + alpha2 * direction)
    
    # Iteratively narrow the bracket using the golden ratio
    while abs(alpha_high - alpha_low) > tol:
        if f1 < f2:
            # Minimum is in the lower segment
            alpha_high = alpha2
            alpha2 = alpha1
            f2 = f1
            alpha1 = alpha_low + golden_section * (alpha_high - alpha_low)
            f1 = f(x + alpha1 * direction)
        else:
            # Minimum is in the upper segment
            alpha_low = alpha1
            alpha1 = alpha2
            f1 = f2
            alpha2 = alpha_high - golden_section * (alpha_high - alpha_low)
            f2 = f(x + alpha2 * direction)
            
    return (alpha_low + alpha_high) / 2  # Return midpoint of final bracket

def powell(f, x0, bounds=None, tol=1e-13, ftol=1e-13, max_iter=10000):
    """
    Implements Powell's method for multidimensional optimization with enhanced features.
    Powell's method minimizes a function by performing sequential line minimizations
    along a set of directions that are updated iteratively.
    
    Parameters:
    -----------
    f : callable
        Objective function to minimize
    x0 : ndarray
        Initial guess
    bounds : list of tuples, optional
        List of (min, max) pairs for each variable
    tol : float, optional (default=1e-13)
        Tolerance for coordinate-wise convergence
    ftol : float, optional (default=1e-13)
        Tolerance for function value convergence
    max_iter : int, optional (default=10000)
        Maximum number of iterations
        
    Returns:
    --------
    tuple
        (x_best, f_hist) where x_best is the best solution found and
        f_hist is the history of function values
    """
    n = len(x0)
    x = np.array(x0, dtype=np.float64)  # Higher precision for better accuracy
    directions = np.eye(n)  # Start with unit vectors as initial directions
    
    def apply_bounds(x):
        """Helper function to enforce bounds on solutions"""
        if bounds is not None:
            return np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
        return x
    
    x = apply_bounds(x)
    f_current = f(x)
    f_best = f_current
    x_best = x.copy()
    
    # Initialize history tracking
    f_hist = [f_current]
    x_hist = [x.copy()]
    
    for iteration in range(max_iter):
        x_start = x.copy()  # Store starting point of iteration
        f_start = f_current
        delta = 0.0  # Track maximum improvement in any direction
        
        # Track improvements for each direction to later sort them
        improvements = []
        
        # Minimize along each direction
        for i in range(n):
            x_prev = x.copy()
            f_prev = f(x_prev)
            
            # Perform line minimization with higher precision
            alpha = line_search(f, x, directions[i], tol=tol/10)
            x_new = x + alpha * directions[i]
            x_new = apply_bounds(x_new)
            f_new = f(x_new)
            
            # Track improvement for direction sorting
            improvement = f_prev - f_new
            improvements.append((improvement, i))
            
            if f_new < f_current:
                x = x_new
                f_current = f_new
                
                # Update best solution if we found a better one
                if f_current < f_best:
                    f_best = f_current
                    x_best = x.copy()
            
            delta = max(delta, abs(f_prev - f(x)))
        
        # Sort directions by their effectiveness (most improvement first)
        improvements.sort(reverse=True)
        new_directions = np.zeros_like(directions)
        for j, (_, idx) in enumerate(improvements):
            new_directions[j] = directions[idx]
        directions = new_directions
        
        # Construct and add new direction based on overall movement
        new_dir = x - x_start
        if np.any(new_dir):
            new_dir = new_dir / np.linalg.norm(new_dir)
            # Replace least effective direction with new direction
            directions[-1] = new_dir
        
        # Orthogonalize direction set using modified Gram-Schmidt
        # This helps maintain numerical stability and search efficiency
        for i in range(n):
            for j in range(i):
                directions[i] -= np.dot(directions[i], directions[j]) * directions[j]
            norm = np.linalg.norm(directions[i])
            if norm > 1e-10:  # Only normalize significant directions
                directions[i] /= norm
        
        # Update history
        f_hist.append(f_current)
        x_hist.append(x.copy())
        
        # Check convergence based on recent improvement
        if len(f_hist) >= 3:
            recent_improvement = abs(f_hist[-3] - f_hist[-1])
            if recent_improvement < ftol:
                break
        
        # Check convergence based on coordinate changes
        if np.all(abs(x - x_start) < tol):
            break
    
    return x_best, f_hist