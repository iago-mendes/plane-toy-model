import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import sys

colors = [
  '#4c72b0', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c',
  '#ccb974', '#64b5cd'
]
plt.rcParams.update({
	'text.usetex' : True,
	'font.family' : 'serif',
	'font.serif' : ['Computer Modern Serif'],
	'font.size': 15,
  'axes.prop_cycle': cycler('color', colors)
})


def run_relaxation(N_values, domain_start, boundary_condition_start, boundary_condition_end, initial_guess, test_id = ''):
  print('#########################################')
  print(test_id)
  fig, axes = plt.subplots(2, 1, squeeze=True)
  for N in N_values:
    print(f'\tN = {N}')
    # Define grid
    r_values = np.linspace(domain_start, 0.1, N)
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

    # Define polar coordinates of flat plane geometry
    A = 0.8 # "stretchiness"
    rho_over_r = np.sqrt(1./A**2 * np.cos(phi)**2 + A**2 * np.sin(phi)**2)
    varphi = np.arctan2(A**2 * np.sin(phi), np.cos(phi))

    # Analytic (conformal) dyad solution
    analytic_e_sigma = np.array([ rho_over_r * np.cos(varphi), rho_over_r * np.sin(varphi) ])
    analytic_e_phi = np.array([ - 1./rho_over_r * np.sin(varphi), 1./rho_over_r * np.cos(varphi) ])

    for xy in range(2):
      if boundary_condition_start == 'analytic':
        conformal_e_sigma[xy,0] = analytic_e_sigma[xy,0]
        conformal_e_phi[xy,0] = analytic_e_phi[xy,0]
      if boundary_condition_end == 'analytic':
        conformal_e_sigma[xy,-1] = analytic_e_sigma[xy,-1]
        conformal_e_phi[xy,-1] = analytic_e_phi[xy,-1]

    if initial_guess == 'analytic':
      conformal_e_sigma = analytic_e_sigma
      conformal_e_phi = analytic_e_phi

    # Evolution parameters
    final_time = 300.
    dt = 0.01 * (10/N)**2
    max_iterations = int(final_time / dt)
    relaxation_parameter = 0.01
    damping_parameter = 1.

    initial_e_sigma = conformal_e_sigma.copy()
    initial_e_phi = conformal_e_phi.copy()

    # Create plots
    def plot_physical_dyad_vectors(iteration=0, show=False, commutator=None, constraint=None):
      physical_e_r = conformal_e_sigma
      physical_e_phi = Omega * conformal_e_phi

      physical_initial_e_r = initial_e_sigma
      physical_initial_e_phi = Omega * initial_e_phi

      fig, axes = plt.subplots(1, 2, squeeze=True)

      axes[0].quiver(x, y, physical_initial_e_r[0], physical_initial_e_r[1], color='black', alpha=0.5)
      axes[1].quiver(x, y, physical_initial_e_phi[0], physical_initial_e_phi[1], color='black', alpha=0.5)

      axes[0].quiver(x, y, physical_e_r[0], physical_e_r[1], color='black')
      axes[1].quiver(x, y, physical_e_phi[0], physical_e_phi[1], color='black')

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

      fig.savefig(f'iterations/physical_dyad_vectors-{iteration:06}.png', format='png', dpi=100)

      if show:
        plt.show()
      
      plt.close()
    def plot_physical_commutator(commutator, iteration=0, show=False, commutator_norm=None):
      fig, axes = plt.subplots(1, 2, squeeze=True)

      im = axes[0].imshow(commutator[0], extent=[phi_values[0], phi_values[-1], r_values[0], r_values[-1]], cmap='viridis', aspect='auto')
      axes[1].imshow(commutator[1], extent=[phi_values[0], phi_values[-1], r_values[0], r_values[-1]], cmap='viridis', aspect='auto')

      axes[0].set_title(r'$[d\vec e^P_{r\phi}]^x$')
      axes[1].set_title(r'$[d\vec e^P_{r\phi}]^y$')

      fig.colorbar(im, ax=axes, orientation='vertical')

      for ax in axes:
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$r$')

      if commutator_norm is not None:
        plt.suptitle(f'relaxation time = {iteration*dt:.1f}, physical commutator vector_norm = {commutator_norm:.2e}')

      fig.set_size_inches(15, 8)

      fig.savefig(f'iterations/physical_commutator-{iteration:06}.png', format='png', dpi=100)

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
      return np.sqrt(integrate(sqr_norm))

    # Metric components
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
    def cross_z(a, b):
      return a[0] * b[1] - a[1] * b[0]
    def cross_xy(w, a):
      return np.array([ - w * a[1], w * a[0] ])
    def angular_velocity(): # out of plane
      result = (
        cross_z(conformal_e_sigma, - partial_phi(conformal_commutator()) - partial_phi(conformal_e_phi))
        + cross_z(conformal_e_phi, Omega * partial_r(conformal_commutator()) + partial_phi(conformal_e_sigma) - conformal_e_phi)
      )
      if boundary_condition_start == 'integration':
        result[0] = cross_z(conformal_e_phi, - conformal_commutator() - conformal_e_phi)[0]
      return result
    def dot_e_sigma(omega, C_ss, C_sp):
      result = relaxation_parameter * cross_xy(omega, conformal_e_sigma) - damping_parameter * (C_ss * conformal_e_sigma + C_sp * conformal_e_phi)
      for xy in range(2):
        if boundary_condition_start == 'analytic':
          result[xy,0] = np.zeros_like(result[xy,0])
        if boundary_condition_end == 'analytic':
          result[xy,-1] = np.zeros_like(result[xy,-1])
      return result
    def dot_e_phi(omega, C_sp, C_pp):
      result = relaxation_parameter * cross_xy(omega, conformal_e_phi)  - damping_parameter * (C_sp * conformal_e_sigma + C_pp * conformal_e_phi)
      for xy in range(2):
        if boundary_condition_start == 'analytic':
          result[xy,0] = np.zeros_like(result[xy,0])
        if boundary_condition_end == 'analytic':
          result[xy,-1] = np.zeros_like(result[xy,-1])
      return result

    # Evolution
    relaxation_time = []
    commutator_residuals = []
    constraint_residuals = []
    for i in range(int(max_iterations)):
      constraint_ss = np.array([C_ss()] * 2)
      constraint_sp = np.array([C_sp()] * 2)
      constraint_pp = np.array([C_pp()] * 2)

      if i % 10 == 0:
        commutator = physical_commutator()
        commutator_residual = vector_norm(commutator)
        constraint_residual = tensor_norm(constraint_ss[0], constraint_sp[0], constraint_pp[0])

        if (np.isnan(commutator_residual) or np.isnan(constraint_residual)):
          print(f'\t\tError: NaN values!')
          break

        relaxation_time.append(i * dt)
        commutator_residuals.append(commutator_residual)
        constraint_residuals.append(constraint_residual)

        if i % 1000 == 0:
          print(f'\t\t{i*dt:.3f}: commutator = {commutator_residual:.2e}, constraint = {constraint_residual:.2e}')

        # if i % 1000 == 0 and i * dt < 10:
        # if i % 1000 == 0:
        #   plot_physical_dyad_vectors(iteration=i, commutator=commutator_residual, constraint=constraint_residual)
        #   plot_physical_commutator(commutator, iteration=i, commutator_norm=commutator_residual)
      
      omega = angular_velocity()
      conformal_e_sigma += dt * dot_e_sigma(omega, constraint_ss, constraint_sp)
      conformal_e_phi += dt * dot_e_phi(omega, constraint_sp, constraint_pp)

    # Plot residuals
    axes[0].plot(relaxation_time, commutator_residuals, label=f'N = {N}')
    axes[1].plot(relaxation_time, constraint_residuals, label=f'N = {N}')
  
  axes[0].set_title('Residuals')
  axes[1].set_xlabel('Relaxation time')
  axes[0].set_ylabel('Commutator residual')
  axes[1].set_ylabel('Constraint residual')
  axes[0].set_xticklabels([])
  for ax in axes:
    ax.set_yscale('log')
    ax.legend()
    ax.grid('on', linestyle='--', alpha=0.5)
  fig.set_size_inches(16,8)
  plt.tight_layout()
  plt.subplots_adjust(hspace=0.025)
  fig.savefig(f'residuals-{test_id}.png', format='png', bbox_inches='tight')

  print()

if 'test1' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0.001, 'none', 'none', 'flat', test_id='test1')
if 'test2' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0.001, 'analytic', 'analytic', 'flat', test_id='test2')
if 'test3' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0., 'analytic', 'analytic', 'flat', test_id='test3')
if 'test4' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0., 'integration', 'analytic', 'flat', test_id='test4')
if 'test5' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0., 'integration', 'none', 'flat', test_id='test5')
if 'test6' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0.001, 'none', 'none', 'analytic', test_id='test6')
if 'test7' in sys.argv:
  run_relaxation([10, 20, 50, 100], 0., 'integration', 'none', 'analytic', test_id='test7')
