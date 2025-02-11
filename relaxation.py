import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex' : True,
	'font.family' : 'serif',
	'font.serif' : ['Computer Modern Serif'],
	'font.size': 15,
})

# Define grid
N = 10
r = np.linspace(0.1, 10., N)
phi = np.linspace(0., 2. * np.pi, 2 * N, endpoint=False)
dr = r[1] - r[0]
dphi = phi[1] - phi[0]
r, phi = np.meshgrid(r, phi, indexing='ij')
sigma = np.log(r)

# Ploting grid
x = r * np.cos(phi)
y = r * np.sin(phi)

# Conformal factor (as a 2D vector)
Omega = np.array([r] * 2)

# Initialize dyad vectors
conformal_e_sigma = np.array([ np.cos(phi), np.sin(phi) ])
conformal_e_phi = np.array([ -np.sin(phi), np.cos(phi) ])

# Evolution parameters
dt = 1e-2
max_iterations = 1e3

# Plot physical dyad vectors
def plot_physical_dyad_vectors(iteration=0, show=False, commutator_norm=None):
  physical_e_r = conformal_e_sigma
  physical_e_phi = Omega * conformal_e_phi

  fig, axes = plt.subplots(1, 2, squeeze=True)

  axes[0].quiver(x, y, physical_e_r[0], physical_e_r[1])
  axes[1].quiver(x, y, physical_e_phi[0], physical_e_phi[1])

  axes[0].set_title(r'$\vec e^P_r$')
  axes[1].set_title(r'$\vec e^P_\phi$')

  for ax in axes:
    ax.set_xlabel(r'$r$')
    ax.set_aspect('equal')
    ax.grid('on', linestyle='--', alpha=0.5)
  axes[0].set_ylabel(r'$\phi$')
  axes[1].set_yticklabels([])

  if commutator_norm is not None:
    plt.suptitle(f'relaxation time = {iteration*dt:.1f}, physical commutator norm = {commutator_norm:.2e}')

  fig.set_size_inches(15, 8)

  fig.savefig(f'iterations/{iteration:04}.png', format='png', dpi=100)

  if show:
    plt.show()
  
  plt.close()

# Compute partial derivatives of 2D vector fields
def partial_r(vector_field):
  return np.gradient(vector_field, dr, axis=1)
def partial_phi(vector_field):
  return (-1./2. * np.roll(vector_field, -1, axis=2) + 1./2. * np.roll(vector_field, 1, axis=2)) / dphi

# Compute commutators
def conformal_commutator():
  return Omega * partial_r(conformal_e_phi) - partial_phi(conformal_e_sigma)
def physical_commutator():
  return Omega**2 / 2. * partial_r(conformal_e_phi) - Omega / 2. * partial_phi(conformal_e_sigma) + Omega / 2. * conformal_e_phi

# Compute dyad updates
def dot_e_sigma():
  return - partial_phi(conformal_commutator()) + partial_phi(conformal_e_phi)
def dot_e_phi():
  return Omega * partial_r(conformal_commutator()) - partial_phi(conformal_e_sigma) + conformal_e_phi

# Evolution
for i in range(int(max_iterations)):
  if i % 10 == 0:
    commutator_norm = np.linalg.norm(physical_commutator())
    print(f'{i}: physical commutator norm = {commutator_norm:.2e}')
    plot_physical_dyad_vectors(iteration=i, commutator_norm=commutator_norm)

  conformal_e_sigma += dt * dot_e_sigma()
  conformal_e_phi += dt * dot_e_phi()
