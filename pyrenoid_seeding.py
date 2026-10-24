import numpy as np
import scipy.fft as sfft
import os
import time
np.random.seed(0)
from params import *


# Grid
Lx = Nx * dx
Ly = Ny * dx
Lz = Nz * dx
x = np.arange(Nx) * dx # grid points
y = np.arange(Ny) * dx
z = np.arange(Nz) * dx
X, Y, Z = np.meshgrid(x, y, z, indexing="ij") # grid
cx, cy, cz = Lx/2, Ly/2, Lz/2 # center of the domain
shape = (Nx, Ny, Nz)

# Time parameters
t_final = t_flow # seeding runs until t_flow
steps = round(t_final / dt)

# initial condition, seeding
r = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) # distance from the center
R0 = R_seed # radius of the seed
phi_U = np.where(r < R0, 0.9, 0.08) + 0.01 * np.random.randn(*shape)
phi_P = np.full(shape, 1e-3)

# Helper functions
def fft3(a):
    return sfft.rfftn(a, workers=W)

def ifft3(a):
    return sfft.irfftn(a, s=shape, axes=(0, 1, 2), workers=W)


# Wavenumber grid
kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dx) * 2 * np.pi
kz = np.fft.rfftfreq(Nz, d=dx) * 2 * np.pi
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
k2 = kx**2 + ky**2 + kz**2
k4 = k2**2

kx1 = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky1 = np.fft.fftfreq(Ny, d=dx) * 2 * np.pi
kz1 = np.fft.rfftfreq(Nz, d=dx) * 2 * np.pi
kx1_d = kx1.copy(); kx1_d[Nx // 2] = 0
ky1_d = ky1.copy(); ky1_d[Ny // 2] = 0
kz1_d = kz1.copy(); kz1_d[-1] = 0
kx_d, ky_d, kz_d = np.meshgrid(kx1_d, ky1_d, kz1_d, indexing="ij")

#### Functions
# Gradient: Fourier-space input, real-space output
def grad(a_tilde):
    return (ifft3(1j * kx_d * a_tilde),
            ifft3(1j * ky_d * a_tilde),
            ifft3(1j * kz_d * a_tilde))

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

# Total divergence term (flux + noise, no flow during seeding)
def div_total(phi, g):
    gx, gy, gz = g
    amp = np.sqrt(2 * noise * Lambda * phi / (dt * dx**3))
    Jx = Lambda * phi * gx + amp * np.random.standard_normal(shape)
    Jy = Lambda * phi * gy + amp * np.random.standard_normal(shape)
    Jz = Lambda * phi * gz + amp * np.random.standard_normal(shape)
    return (1j * kx_d * fft3(Jx)
          + 1j * ky_d * fft3(Jy)
          + 1j * kz_d * fft3(Jz))

# time formatting
def hms(sec):
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"

# stabilization
L_U = Lambda * (A_U * k2 + kappa * k4)
L_P = Lambda * A_P * k2

# denominators
denom_U = 1 + dt * L_U
denom_P = 1 + dt * L_P

# saving frames and times for plotting
os.makedirs("frames", exist_ok=True)

np.savez("frames/params.npz", Nx=Nx, Ny=Ny, Nz=Nz, dx=dx, dt=dt, t_settle=t_settle, t_flow=t_flow, t_final=t_final,
         chi=chi, kappa=kappa, Lambda=Lambda, b=b, phi_c=phi_c,
         k_P=k_P, k_U=k_U, R0=R0, cx=cx, cy=cy, cz=cz, x=x, y=y, z=z)

def save_frame(i, t):
    np.savez(f"frames/frame_{i:04d}.npz",
             U=phi_U.astype(np.float32), P=phi_P.astype(np.float32), t=t)


# spectral method time-stepping loop
frame_idx = 0
save_frame(frame_idx, 0.0)
report_every = max(1, steps // 1000)
t0 = time.perf_counter()


for step in range(steps):
    # delaying reaction
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

    # gradients
    gU = grad(mu_U_tilde)
    gP = grad(mu_P_tilde)
    # source term
    s_U_tilde = fft3(s(phi_U, phi_P))

    N_U = div_total(phi_U, gU) + react * s_U_tilde
    N_P = div_total(phi_P, gP) - react * s_U_tilde

    phi_U_tilde = (phi_U_tilde + dt * N_U + dt * L_U * phi_U_tilde) / denom_U
    phi_P_tilde = (phi_P_tilde + dt * N_P + dt * L_P * phi_P_tilde) / denom_P

    phi_U = ifft3(phi_U_tilde)
    phi_P = ifft3(phi_P_tilde)

    # saving frames
    if (step + 1) % save_interval == 0:
        frame_idx += 1
        save_frame(frame_idx, (step + 1) * dt)

    # status update
    if step % report_every == 0:
        done = step + 1
        elapsed = time.perf_counter() - t0
        remaining = elapsed / done * (steps - done)
        print(f"{100*done/steps:5.1f}%  t = {t:.1f}  "
              f"elapsed {hms(elapsed)}  left {hms(remaining)}", flush=True)

        # stopping if NaN appears
        if np.isnan(phi_U).any() or np.isnan(phi_P).any():
            print(f"NaN at t = {t:.2f}, stopping")
            break

    # saving the state at t_flow for later use
    if step + 1 == round(t_flow / dt):
        np.save("state_U_tflow.npy", phi_U)
        np.save("state_P_tflow.npy", phi_P)
