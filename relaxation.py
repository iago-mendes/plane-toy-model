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
r_values = np.linspace(0.1, 10., N)
phi_values = np.linspace(0., 2. * np.pi, 2 * N, endpoint=False)
dr = r_values[1] - r_values[0]
dphi = phi_values[1] - phi_values[0]
r, phi = np.meshgrid(r_values, phi_values, indexing='ij')

# Ploting grid
x = r * np.cos(phi)
y = r * np.sin(phi)

# Conformal factor (as a 2D vector)
Omega = np.array([r] * 2)

# Initialize dyad vectors
conformal_e_sigma = np.array([ np.cos(phi), np.sin(phi) ])
conformal_e_phi = np.array([ -np.sin(phi), np.cos(phi) ])

# Evolution parameters
dt = 0.01 * (10/N)**2
# max_iterations = 1
max_iterations = int(200. / dt)
relaxation_parameter = 0.
damping_parameter = 1.

# Create plots
def plot_physical_dyad_vectors(iteration=0, show=False, commutator=None, constraint=None):
  physical_e_r = conformal_e_sigma
  physical_e_phi = Omega * conformal_e_phi

  fig, axes = plt.subplots(1, 2, squeeze=True)

  axes[0].quiver(x, y, physical_e_r[0], physical_e_r[1])
  axes[1].quiver(x, y, physical_e_phi[0], physical_e_phi[1])

  axes[0].set_title(r'$\vec e^P_r$')
  axes[1].set_title(r'$\vec e^P_\phi$')

  for ax in axes:
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    ax.grid('on', linestyle='--', alpha=0.5)

  if commutator is not None and constraint is not None:
    plt.suptitle(f'relaxation time = {iteration*dt:.1f}, commutator = {commutator:.2e}, constraint = {constraint:.2e}')

  fig.set_size_inches(15, 8)

  fig.savefig(f'iterations/physical_dyad_vectors-{iteration:04}.png', format='png', dpi=100)

  if show:
    plt.show()
  
  plt.close()
def plot_physical_commutator(commutator, iteration=0, show=False, commutator_norm=None):
  fig, axes = plt.subplots(1, 2, squeeze=True)

  im = axes[0].imshow(commutator[0], extent=[phi_values[0], phi_values[-1], r_values[0], r_values[-1]], cmap='viridis')
  axes[1].imshow(commutator[1], extent=[phi_values[0], phi_values[-1], r_values[0], r_values[-1]], cmap='viridis')

  axes[0].set_title(r'$[d\vec e^P_{r\phi}]^x$')
  axes[1].set_title(r'$[d\vec e^P_{r\phi}]^y$')

  fig.colorbar(im, ax=axes, orientation='vertical')

  for ax in axes:
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$r$')

  if commutator_norm is not None:
    plt.suptitle(f'relaxation time = {iteration*dt:.1f}, physical commutator vector_norm = {commutator_norm:.2e}')

  fig.set_size_inches(15, 8)

  fig.savefig(f'iterations/physical_commutator-{iteration:04}.png', format='png', dpi=100)

  if show:
    plt.show()
  
  plt.close()

# Compute partial derivatives of 2D vector fields
def partial_r(vector_field):
  # Centered FD
  df = 1./2. * np.roll(vector_field, -1, axis=1) - 1./2. * np.roll(vector_field, 1, axis=1)
  for xy in range(2):
    for j in range(2*N):
      # Forward FD
      df[xy,0,j] = -1. * vector_field[xy,0,j] + 1. * vector_field[xy,1,j]
      # Backward FD
      df[xy,-1,j] = -1. * vector_field[xy,-2,j] + 1. * vector_field[xy,-1,j]
  return df / dr
def partial_phi(vector_field):
  df = 1./2. * np.roll(vector_field, -1, axis=2) - 1./2. * np.roll(vector_field, 1, axis=2)
  return df / dphi

# Compute commutators
def conformal_commutator():
  return Omega * partial_r(conformal_e_phi) - partial_phi(conformal_e_sigma)
def physical_commutator():
  return Omega**2 / 2. * partial_r(conformal_e_phi) - Omega / 2. * partial_phi(conformal_e_sigma) + Omega / 2. * conformal_e_phi

# Compute integrals
def integrate(field):
  result = 0.
  for i in range(N):
    dA = r_values[i] * dr * dphi
    for j in range(2*N):
      result += field[i,j] * dA
  return result
def vector_norm(vector_field): # rank 1
  sqr_norm = vector_field[0]**2 + r**2 * vector_field[1]**2
  return np.sqrt(integrate(sqr_norm))
def tensor_norm(field_ss, field_sp, field_pp): # rank 2
  sqr_norm = r**4 * (field_ss**2 + 2. * field_sp**2 + field_pp**2)
  # print(sqr_norm)
  return np.sqrt(integrate(sqr_norm))

# Metric components
A = 0.6 # "stretchiness"
def g_ss():
  # return np.sqrt(1. / A**2 * np.cos(phi)**2 + A**2 * np.sin(phi)**2)
  return r**2 * np.sqrt(1. / A**2 * np.cos(phi)**2 + A**2 * np.sin(phi)**2)
def g_sp():
  # return r * (A**2 - 1. / A**2) * np.sin(phi) * np.cos(phi)
  return r**2 * (A**2 - 1. / A**2) * np.sin(phi) * np.cos(phi)
def g_pp():
  return r**2 / (1. / A**2 * np.cos(phi)**2 + A**2 * np.sin(phi)**2) * ( 1. + (A**2 - 1. / A**2)**2 * np.sin(phi)**2 * np.cos(phi)**2)

# Constraints
def C_ss():
  return r**2 * (conformal_e_sigma[0] * conformal_e_sigma[0] + conformal_e_sigma[1] * conformal_e_sigma[1]) - g_ss()
def C_sp():
  return r**2 * (conformal_e_sigma[0] * conformal_e_phi[0] + conformal_e_sigma[1] * conformal_e_phi[1]) - g_sp()
def C_pp():
  return r**2 * (conformal_e_phi[0] * conformal_e_phi[0] + conformal_e_phi[1] * conformal_e_phi[1]) - g_pp()

# Dyad updates
def dot_e_sigma(C_ss, C_sp):
  return relaxation_parameter * (- partial_phi(conformal_commutator()) - partial_phi(conformal_e_phi)) - damping_parameter / Omega**2 * (C_ss * conformal_e_sigma + C_sp * conformal_e_phi)
  # return - damping_parameter / Omega**2 * (C_ss * conformal_e_sigma + C_sp * conformal_e_phi)
def dot_e_phi(C_sp, C_pp):
  return relaxation_parameter * (Omega * partial_r(conformal_commutator()) + partial_phi(conformal_e_sigma) - conformal_e_phi)  - damping_parameter / Omega**2 * (C_sp * conformal_e_sigma + C_pp * conformal_e_phi)
  # return - damping_parameter / Omega**2 * (C_sp * conformal_e_sigma + C_pp * conformal_e_phi)

# Evolution
relaxation_time = []
commutator_residuals = []
constraint_residuals = []
for i in range(int(max_iterations)):
  constraint_ss = np.array([C_ss()] * 2)
  constraint_sp = np.array([C_sp()] * 2)
  constraint_pp = np.array([C_pp()] * 2)

  if (i*dt) > 100.:
    relaxation_parameter = 1e-5
    damping_parameter = 1e-1
  if (i*dt) > 110.:
    relaxation_parameter = 1e-4
    damping_parameter = 1e-2
  if (i*dt) > 120.:
    relaxation_parameter = 1e-3
    damping_parameter = 1e-3
  if (i*dt) > 130.:
    relaxation_parameter = 1e-2
    damping_parameter = 1e-4
  if (i*dt) > 140.:
    relaxation_parameter = 1e-1
    damping_parameter = 1e-5
  if (i*dt) > 150.:
    relaxation_parameter = 1.
    damping_parameter = 0.
  
  if i % 10 == 0:
    relaxation_time.append(i * dt)

    commutator = physical_commutator()
    commutator_residual = vector_norm(commutator)
    commutator_residuals.append(commutator_residual)

    constraint_residual = tensor_norm(constraint_ss[0], constraint_sp[0], constraint_pp[0])
    constraint_residuals.append(constraint_residual)

    print(f'{i*dt:.3f}: commutator = {commutator_residual:.2e}, constraint = {constraint_residual:.2e}')

    # plot_physical_dyad_vectors(iteration=i, commutator=commutator_residual, constraint=constraint_residual)
    # plot_physical_commutator(commutator, iteration=i, commutator_norm=commutator_residual)
  
  conformal_e_sigma += dt * dot_e_sigma(constraint_ss, constraint_sp)
  conformal_e_phi += dt * dot_e_phi(constraint_sp, constraint_pp)

# Plot residuals
fig, ax = plt.subplots(1, 1, squeeze=True)
ax.set_title('Residuals')
ax.plot(relaxation_time, commutator_residuals, label='Commutator')
ax.plot(relaxation_time, constraint_residuals, label='Constraint')
ax.set_xlabel('Relaxation time')
ax.set_yscale('log')
ax.legend()
ax.grid('on', linestyle='--', alpha=0.5)
fig.set_size_inches(16,8)
plt.tight_layout()
fig.savefig(f'residuals.png', format='png', bbox_inches='tight')
plt.show()
