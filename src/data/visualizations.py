import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import simulate

# Run one simulation
rng        = np.random.default_rng(seed=42)
trajectory = simulate(rng=rng)          # shape: (101, 64, 64)

# Option 1: Save as MP4 animation (best for professor)
fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('black')
ax.axis('off')

# imshow sets up the heatmap, we'll update it each frame
im = ax.imshow(
    trajectory[0],
    cmap='inferno',        # black→red→yellow→white (looks like heat)
    vmin=0, vmax=1,        # fixed scale so colors don't jump between frames
    interpolation='bilinear'
)

title = ax.set_title('t = 0', color='white', fontsize=13, pad=8)
plt.colorbar(im, ax=ax, fraction=0.046, label='Temperature')

def update(frame):
    im.set_data(trajectory[frame])
    title.set_text(f't = {frame}')
    return [im, title]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(trajectory),   # 101 frames
    interval=60,              # ms between frames (~16 fps)
    blit=True
)

# Save as GIF (no extra install needed)
ani.save('heat_simulation.gif', writer='pillow', fps=16, dpi=120)
print("Saved: heat_simulation.gif")

plt.close()

# Option 2: Static 9-panel snapshot grid
# Shows t=0, 10, 20, ... 80 in one image
snapshots   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
fig2, axes  = plt.subplots(2, 5, figsize=(15, 6))
fig2.patch.set_facecolor('black')
fig2.suptitle('Heat Diffusion — 64×64 Grid', color='white', fontsize=14, y=1.01)

for ax, t in zip(axes.flat, snapshots):
    ax.imshow(trajectory[t], cmap='inferno', vmin=0, vmax=1)
    ax.set_title(f't = {t}', color='white', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('heat_snapshots.png', dpi=150, bbox_inches='tight',
            facecolor='black')
print("Saved: heat_snapshots.png")
plt.close()