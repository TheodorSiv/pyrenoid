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
t_flow = t_settle + 100
t_final = t_flow + 200

dt = 0.001
steps = int(t_final / dt)
save_interval = 2000 

# Parameters
chi = 5 # Flory-Huggins parameter
Lambda = 1 # mobility
kappa = 1 # gradient coefficient
b = 200
phi_c = 0.5
k_P, k_U = 0.03, 0.03
eps = 12.5

# eta_in = 
# eta_out =

alpha = 1000 

eta = 40

# grid
x = np.arange(N) * dx # grid points
X, Y, Z = np.meshgrid(x, x, x, indexing="ij") # grid
c = L / 2 # center of the domain

# loading initial conditions
phi_U = np.load("state_U_tflow.npy")
phi_P = np.load("state_P_tflow.npy")

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
k2_d = kx_d**2 + ky_d**2 + kz_d**2
k2_safe = np.where(k2_d == 0, 1, k2_d)

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

# Divergence of flux term
def div_flux(phi, mu_tilde):
    # gradient of mu in real space
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

# Divergence of advection term
def div_adv(phi, vx, vy, vz):
    return (1j * kx_d * fft3(vx * phi)
            + 1j * ky_d * fft3(vy * phi)
            + 1j * kz_d * fft3(vz * phi))

# Korteweg force term
def korteweg(phi, mu_tilde):
    return (-phi * ifft3(1j * kx_d * mu_tilde),
            -phi * ifft3(1j * ky_d * mu_tilde),
            -phi * ifft3(1j * kz_d * mu_tilde))

# Noise term
noise = 10**(-5)
def div_noise(phi):
    amp = np.sqrt(2 * noise * Lambda * phi / (dt * dx**3))
    return (1j * kx_d * fft3(amp * np.random.standard_normal(shape))
          + 1j * ky_d * fft3(amp * np.random.standard_normal(shape))
          + 1j * kz_d * fft3(amp * np.random.standard_normal(shape)))

# Solving the Cahn-Hilliard equation using spectral method
# stabilization
A_U, A_P = 100, 5
L_U = Lambda * (A_U * k2 + kappa * k4)
L_P = Lambda * A_P * k2
denom_U = 1 + dt * L_U
denom_P = 1 + dt * L_P

# external force
std = 1.2
F0 = 36
R0 = 7.2 + 0.1       # from seeding file, update at next seed 
r = np.sqrt((Y - c)**2 + (Z - c)**2)
g = np.exp(- (X - c)**2 / (2 * std**2)) * np.exp(- (r - R0)**2 / (2 * std**2))
eps0 = 1e-8
fx = np.zeros_like(X)
fy = - F0 * g * (Y - c) / (r + eps0)
fz = - F0 * g * (Z - c) / (r + eps0)

# flow
fx_tilde = fft3(fx)
fy_tilde = fft3(fy)
fz_tilde = fft3(fz)

# denominator for velocity field
denom_v = eta * k2 + eps

# flow from Gaussian pressure field
#P = 10 * np.exp(-(X - c)**2 / (2 * 1.2**2))
#vx = ifft3(-1j * kx_d * fft3(P) / (eta * k2 + eps))
#vy = np.zeros(shape)
#vz = np.zeros(shape)

# saving frames and times for plotting
os.makedirs("frames", exist_ok=True)

np.savez("frames/params.npz", N=N, dx=dx, dt=dt, t_settle=t_settle, t_flow=t_flow, t_final=t_final,
         chi=chi, kappa=kappa, Lambda=Lambda, b=b, phi_c=phi_c,
         k_P=k_P, k_U=k_U, eta=eta, eps=eps, F0=F0, std=std, c=c, x=x, R0=R0)

def save_frame(i, t, vx, vy, vz):
    np.savez(f"frames/frame_{i:04d}.npz",
             U=phi_U.astype(np.float32), P=phi_P.astype(np.float32), t=t,
             vx=vx.astype(np.float32), vy=vy.astype(np.float32), vz=vz.astype(np.float32))


# spectral method time-stepping loop
start_step = round(t_flow / dt)
frame_idx = start_step // save_interval
# save_frame(frame_idx, t_flow)

for step in range(start_step, steps):
    # delaying raction and flow
    t = step * dt
    react = 1.0 if t >= t_settle else 0.0
    flow = 1.0 if t >= t_flow else 0.0
    
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

    KUx, KUy, KUz = korteweg(phi_U, mu_U_tilde)
    KPx, KPy, KPz = korteweg(phi_P, mu_P_tilde)
    Fx_tilde = fx_tilde + alpha * fft3(KUx + KPx)
    Fy_tilde = fy_tilde + alpha *fft3(KUy + KPy)
    Fz_tilde = fz_tilde + fft3(KUz + KPz)
    kdotF = (kx_d * Fx_tilde + ky_d * Fy_tilde + kz_d * Fz_tilde) / k2_safe
    vx = ifft3((Fx_tilde - kx_d * kdotF) / denom_v)
    vy = ifft3((Fy_tilde - ky_d * kdotF) / denom_v)
    vz = ifft3((Fz_tilde - kz_d * kdotF) / denom_v)

    s_U_tilde = fft3(s(phi_U, phi_P))

    N_U = div_flux(phi_U, mu_U_tilde) - flow * div_adv(phi_U, vx, vy, vz) + react * s_U_tilde
    N_P = div_flux(phi_P, mu_P_tilde) - flow * div_adv(phi_P, vx, vy, vz) - react * s_U_tilde

    N_U += div_noise(phi_U)
    N_P += div_noise(phi_P)

    phi_U_tilde = (phi_U_tilde + dt * N_U + dt * L_U * phi_U_tilde) / denom_U
    phi_P_tilde = (phi_P_tilde + dt * N_P + dt * L_P * phi_P_tilde) / denom_P

    phi_U = ifft3(phi_U_tilde)
    phi_P = ifft3(phi_P_tilde)

    # saving frames 
    if (step + 1) % save_interval == 0:
        frame_idx += 1
        save_frame(frame_idx, (step + 1) * dt, vx, vy, vz)

    # status update
    if (step - start_step) % max(1, (steps - start_step) // 1000) == 0:
        print(f"Status: {100 * (step - start_step) / (steps - start_step):.1f}%  (t = {t:.1f})")
        
        vmax = np.sqrt(vx**2 + vy**2 + vz**2).max()
        print(f"  max|v| = {vmax:.3g},  CFL = {vmax*dt/dx:.3g}")

        # stopping if NaN appears
        if np.isnan(phi_U).any() or np.isnan(phi_P).any():
            print(f"NaN at t = {t:.2f}, stopping")
            break

