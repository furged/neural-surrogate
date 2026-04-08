import numpy as np

# PARAMETERS (all in one place — easy to change)

GRID_SIZE   = 64       # NxN grid
ALPHA       = 0.1      # diffusion coefficient (how fast heat spreads)
DT          = 0.1      # time step Δt
DX          = 1.0      # spatial step (grid spacing)
NUM_STEPS   = 100      # how many timesteps per simulation
NUM_BLOBS   = 3        # number of Gaussian hot spots in initial condition

# Stability check: for explicit finite difference, need α*Δt/Δx² < 0.25
# With α=0.1, Δt=0.1, Δx=1.0 → 0.1*0.1/1.0 = 0.01 ✓ stable
STABILITY = ALPHA * DT / (DX ** 2)
assert STABILITY < 0.25, f"Unstable! α*Δt/Δx² = {STABILITY:.3f}, must be < 0.25" # if condition after "assert" is false, python simply crashes with error, so this is a mere alarm for safety


# INITIAL CONDITION: Random Gaussian Blobs
def generate_initial_condition(grid_size, num_blobs, rng):
    """
    Creates a temperature field with random Gaussian hot spots.
    
    Each blob is defined by:
      - Random center (cx, cy)
      - Random amplitude (peak temperature)
      - Random spread (sigma)
    
    This gives diverse initial conditions → model learns general dynamics.
    """
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)  # shape: (grid_size, grid_size)

    field = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _ in range(num_blobs):
        cx    = rng.uniform(0, grid_size)          # blob center x
        cy    = rng.uniform(0, grid_size)          # blob center y
        amp   = rng.uniform(0.5, 1.0)              # peak temperature
        sigma = rng.uniform(3.0, 10.0)             # spread

        blob = amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        field += blob

    # Normalize to [0, 1] so model sees consistent scale
    max_val = field.max()
    if max_val > 0:
        field /= max_val

    return field


# FINITE DIFFERENCE STEP: Periodic Boundary
def laplacian_periodic(u, dx):
    """
    Computes the discrete Laplacian ∇²u using finite differences.
    
    Formula (2D):
      ∇²u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx²

    Periodic BC: np.roll wraps the grid automatically.
      - np.roll(u, -1, axis=0) → shifts rows UP   (gives i+1 neighbor)
      - np.roll(u,  1, axis=0) → shifts rows DOWN  (gives i-1 neighbor)
      - same for columns (axis=1)
    
    This means the right edge is a neighbor of the left edge, and
    the top edge is a neighbor of the bottom edge.
    """
    u_ip1 = np.roll(u, -1, axis=0)   # i+1 (next row, wraps)
    u_im1 = np.roll(u,  1, axis=0)   # i-1 (prev row, wraps)
    u_jp1 = np.roll(u, -1, axis=1)   # j+1 (next col, wraps)
    u_jm1 = np.roll(u,  1, axis=1)   # j-1 (prev col, wraps)

    return (u_ip1 + u_im1 + u_jp1 + u_jm1 - 4 * u) / (dx ** 2)


def step(u, alpha, dt, dx):
    """
    Advances the heat equation by one timestep using explicit Euler:

      u(t+Δt) = u(t) + Δt * α * ∇²u(t)

    This is the mapping your CNN will learn to approximate.
    """
    lap = laplacian_periodic(u, dx)
    return u + dt * alpha * lap


# FULL SIMULATION: One trajectory
def simulate(grid_size=GRID_SIZE, num_steps=NUM_STEPS, alpha=ALPHA,
             dt=DT, dx=DX, num_blobs=NUM_BLOBS, rng=None):
    """
    Runs one full simulation from a random initial condition.

    Returns:
        trajectory: np.array of shape (num_steps+1, grid_size, grid_size)
                    trajectory[0]  = initial state
                    trajectory[t]  = state at timestep t
    """
    if rng is None:
        rng = np.random.default_rng()

    u = generate_initial_condition(grid_size, num_blobs, rng)

    trajectory = np.empty((num_steps + 1, grid_size, grid_size), dtype=np.float32)
    trajectory[0] = u

    for t in range(num_steps):
        u = step(u, alpha, dt, dx)
        trajectory[t + 1] = u

    return trajectory

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    print("Running simulator test...")
    traj = simulate(rng=rng)

    print(f"Trajectory shape : {traj.shape}")          # expect (101, 64, 64)
    print(f"Initial max temp : {traj[0].max():.4f}")   # expect ~1.0 (normalized)
    print(f"Final max temp   : {traj[-1].max():.4f}")  # expect < initial (heat diffused)
    print(f"Stability number : {STABILITY:.4f} (must be < 0.25)")
    print("Simulator OK ✓")
