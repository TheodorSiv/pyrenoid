import numpy as np
import scipy.fft as sfft
import os

np.random.seed(0)

# Grid
N = 80
dx = 0.5
L = N*dx

# Time parameters
t_settle = 50
t_flow = t_settle + 150
t_final = t_flow

dt = 0.001
steps = round(t_final / dt)
save_interval = 2000 

# Parameters
chi = 5 # Flory-Huggins parameter
Lambda = 1 # mobility
kappa = 1 # gradient coefficient
b = 200
phi_c = 0.5
k_P, k_U = 0.03, 0.03

# Initial condition, seeding
x = np.arange(N) * dx # grid points
X, Y, Z = np.meshgrid(x, x, x, indexing="ij") # grid
c = L / 2 # center of the domain
r = np.sqrt((X - c)**2 + (Y - c)**2 + (Z - c)**2) # distance from the center
R0 = 7.2 # radius of the seed

phi_U = np.where(r < R0, 0.9, 0.1) + 0.007 * np.random.randn(N, N, N)
phi_P = np.full((N, N, N), 1e-3)

# Helper functions
shape = (N, N, N)
W = 4 # number of cores
def fft3(a):
    return sfft.rfftn(a, workers=W)

def ifft3(a):
    return sfft.irfftn(a, s=shape, axes=(0, 1, 2), workers=W)

# Wavenumber grid
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kz = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
k2 = kx**2 + ky**2 + kz**2
k4 = k2**2

kx1 = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kz1 = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
kx1_d = kx1.copy(); kx1_d[N // 2] = 0
kz1_d = kz1.copy(); kz1_d[-1] = 0 
kx_d, ky_d, kz_d = np.meshgrid(kx1_d, kx1_d, kz1_d, indexing="ij")

# Derivative of the free energy
def df(phi_U, phi_P):
    phi_S = 1 - phi_U - phi_P
    mu_U = np.log(phi_U) - np.log(phi_S) + chi * (phi_S - phi_U)
    mu_P = np.log(phi_P) - np.log(phi_S)
    return mu_U, mu_P

# Reaction flux term
def s(phi_U, phi_P):
    sigma = 1 / (1 + np.exp(- b * (phi_U - phi_c)))
    return - sigma * k_P * phi_U + (1 - sigma) * k_U * phi_P

# Noise term
noise = 10**(-5)
def div_noise(phi):
    amp = np.sqrt(2 * noise * Lambda * phi / (dt * dx**3))
    return (1j * kx_d * fft3(amp * np.random.standard_normal(shape))
          + 1j * ky_d * fft3(amp * np.random.standard_normal(shape))
          + 1j * kz_d * fft3(amp * np.random.standard_normal(shape)))

# Divergence of flux term
def div_flux(phi, mu_tilde):
    # gradient af mu in real space
    dmu_x = ifft3(1j * kx_d * mu_tilde)
    dmu_y = ifft3(1j * ky_d * mu_tilde)
    dmu_z = ifft3(1j * kz_d * mu_tilde)
    # flux in real space
    Jx = Lambda * phi * dmu_x
    Jy = Lambda * phi * dmu_y
    Jz = Lambda * phi * dmu_z

    # divergence in Fourier space
    return (1j * kx_d * fft3(Jx)
          + 1j * ky_d * fft3(Jy)
          + 1j * kz_d * fft3(Jz))

# Solving the Cahn-Hilliard equation using spectral method
# stabilization
A_U, A_P = 100, 5
L_U = Lambda * (A_U * k2 + kappa * k4)
L_P = Lambda * A_P * k2
denom_U = 1 + dt * L_U
denom_P = 1 + dt * L_P

# saving frames and times for plotting
os.makedirs("frames", exist_ok=True)

np.savez("frames/params.npz", N=N, dx=dx, dt=dt, t_settle=t_settle, t_flow=t_flow, t_final=t_final,
         chi=chi, kappa=kappa, Lambda=Lambda, b=b, phi_c=phi_c,
         k_P=k_P, k_U=k_U, R0=R0, c=c, x=x)

def save_frame(i, t):
    np.savez(f"frames/frame_{i:04d}.npz",
             U=phi_U.astype(np.float32), P=phi_P.astype(np.float32), t=t)

frame_idx = 0
save_frame(frame_idx, 0.0)

# spectral method time-stepping loop
for step in range(steps):
    # delaying raction and flow
    t = step * dt
    react = 1.0 if t >= t_settle else 0.0
    
    #clipping
    phi_U = np.clip(phi_U, 1e-8, 1 - 1e-8)
    phi_P = np.clip(phi_P, 1e-8, 1 - 1e-8) 
    tot = phi_U + phi_P
    over = tot > 1 - 1e-8
    scale = (1 - 1e-8) / tot[over]
    phi_U[over] *= scale
    phi_P[over] *= scale
    
    # spectral
    phi_U_tilde = fft3(phi_U)
    phi_P_tilde = fft3(phi_P)

    mu_U, mu_P = df(phi_U, phi_P)
    mu_U_tilde = fft3(mu_U) + kappa * k2 * phi_U_tilde
    mu_P_tilde = fft3(mu_P)

    s_U_tilde = fft3(s(phi_U, phi_P))

    N_U = div_flux(phi_U, mu_U_tilde) + react * s_U_tilde
    N_P = div_flux(phi_P, mu_P_tilde) - react * s_U_tilde

    N_U += div_noise(phi_U)
    N_P += div_noise(phi_P)

    phi_U_tilde = (phi_U_tilde + dt * N_U + dt * L_U * phi_U_tilde) / denom_U
    phi_P_tilde = (phi_P_tilde + dt * N_P + dt * L_P * phi_P_tilde) / denom_P

    phi_U = ifft3(phi_U_tilde)
    phi_P = ifft3(phi_P_tilde)

    # saving frames 
    if (step + 1) % save_interval == 0:
        frame_idx += 1
        save_frame(frame_idx, (step + 1) * dt)

    # status update
    if step % max(1, steps // 1000) == 0:
        print(f"Status: {100 * step / steps:.1f}%")

        # stopping if NaN appears
        if np.isnan(phi_U).any() or np.isnan(phi_P).any():
            print(f"NaN at t = {t:.2f}, stopping")
            break

    # saving the state at t_flow for later use
    if step + 1 == round(t_flow / dt):
        np.save("state_U_tflow.npy", phi_U)
        np.save("state_P_tflow.npy", phi_P)

