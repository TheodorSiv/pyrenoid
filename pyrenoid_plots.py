import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.special import expit
import glob

folder = "frames"

# Parametre
params = np.load(os.path.join(folder, "params.npz"))
dx       = float(params["dx"])
t_settle = float(params["t_settle"])
t_flow   = float(params["t_flow"])
b        = float(params["b"])
phi_c    = float(params["phi_c"])
k_P      = float(params["k_P"])
k_U      = float(params["k_U"])
N        = int(params["N"])
#vx       = params["vx"]
#vy       = params["vy"]
#vz       = params["vz"]
x        = params["x"]
c       = float(params["c"])


# Frames
files = sorted(glob.glob(os.path.join(folder, "frame_*.npz")))
frames_U, frames_P, times = [], [], []
for f in files:
    d = np.load(f)
    frames_U.append(d["U"])
    frames_P.append(d["P"])
    times.append(float(d["t"]))
times = np.array(times)
print(f"Loaded {len(files)} frames, t = {times[0]:.1f} to {times[-1]:.1f}")

# Reaction term (same as in the simulation)
def s(phi_U, phi_P):
    sigma = expit(b * (phi_U - phi_c))
    return - sigma * k_P * phi_U + (1 - sigma) * k_U * phi_P

# Plotting and saving frames with pyvista
os.makedirs("snapshots", exist_ok=True)

def phase(t):
    """Return the name of the simulation phase at time t."""
    if t < t_settle: return "settling"
    if t < t_flow:   return "reactions"
    return "reactions + flow"

# 3D view + 2D slice
pad_shape = np.array(frames_U[0].shape) + 2   # grid shape including padding
half = (pad_shape - 1) * dx / 2
bounds = [-half[0], half[0], -half[1], half[1], -half[2], half[2]]

for i, (fU, t) in enumerate(zip(frames_U, times)):
    data = np.pad(fU, 1, constant_values=fU.min())   # pad boundaries to close the isosurface
    grid = pv.ImageData(dimensions=data.shape, spacing=(dx, dx, dx))
    grid["phi_U"] = data.flatten(order="F")
    p = pv.Plotter(shape=(1, 2), off_screen=True)

    # left: 3D isosurface
    p.subplot(0, 0)
    p.reset_camera(bounds=bounds)
    p.add_mesh(grid.contour([0.5]), color="gold", smooth_shading=True,
               backface_params={"color": "orange"})
    p.show_bounds(bounds=bounds, location="outer")

    # right: 2D slice through the centre
    p.subplot(0, 1)
    p.add_mesh(grid.slice(normal="z"), cmap="cividis", clim=[0, 1])
    p.camera_position = "xy"
    p.add_text(f"t = {t:.1f}  ({phase(t)})", font_size=10)

    p.screenshot(f"snapshots/frame_{i:03d}.png")
    p.close()
    print(f"Frame {i+1}/{len(frames_U)} saved")

times = np.array(times)

# 1. Mass conservation
sum_U = np.array([f.sum() for f in frames_U]) * dx**3
sum_P = np.array([f.sum() for f in frames_P]) * dx**3
total = sum_U + sum_P
plt.figure()
plt.plot(times, sum_U, label="$E_U$")
plt.plot(times, sum_P, label="$E_P$")
plt.plot(times, total, "k--", label="total")
plt.axvline(t_settle, color="gray", ls=":")
plt.axvline(t_flow, color="gray", ls=":")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.title("Mass conservation")
plt.legend()
plt.savefig("snapshots/01_mass.png")
print("Relative drift in total:", (total.max() - total.min()) / total[0])

# 2. Reaction flux inside and outside the droplet
flux_in, flux_out = [], []
for fU, fP, t in zip(frames_U, frames_P, times):
    sv = s(fU, fP) * (t >= t_settle)   # reactions are off during settling
    inside = fU > 0.5
    flux_in.append(sv[inside].sum() * dx**3)
    flux_out.append(sv[~inside].sum() * dx**3)
plt.figure()
plt.plot(times, flux_in, label="inside")
plt.plot(times, flux_out, label="outside")
plt.plot(times, np.array(flux_in) + np.array(flux_out), "k--", label="total")
plt.axvline(t_settle, color="gray", ls=":")
plt.axvline(t_flow, color="gray", ls=":")
plt.xlabel("Time")
plt.ylabel("Reaction flux of $E_U$")
plt.title("Reaction flux")
plt.legend()
plt.savefig("snapshots/02_flux.png")

# 3. Equivalent droplet radius
V = np.array([(f > 0.5).sum() for f in frames_U]) * dx**3
R = (3 * V / (4 * np.pi))**(1/3)
plt.figure()
plt.plot(times, R, label="simulation")
tr = times[times >= t_settle]
plt.plot(tr, R[times >= t_settle][0] * np.exp(-k_P * (tr - t_settle) / 3),
         "k--", label="$R_0e^{-k_Pt/3}$ (valid for $k_U=0$)")
plt.axvline(t_settle, color="gray", ls=":")
plt.axvline(t_flow, color="gray", ls=":")
plt.xlabel("Time")
plt.ylabel("Equivalent radius")
plt.title("Droplet radius")
plt.legend()
plt.savefig("snapshots/03_radius.png")


# 4–6. Velocity field in xy, xz and yz planes, every 5th frame
os.makedirs("snapshots/velocity", exist_ok=True)
i = N // 2
skip = 3 # skip every 3rd point for clarity

for n in range(0, len(files), 5):
    d = np.load(files[n])
    if "vx" not in d.files:
        continue
    vx, vy, vz = d["vx"], d["vy"], d["vz"]
    fU = d["U"]
    t = float(d["t"])

    # xy plane
    plt.figure(figsize=(6, 6))
    plt.quiver(x[::skip], x[::skip], vx[::skip, ::skip, i].T, vy[::skip, ::skip, i].T)
    plt.contour(x, x, fU[:, :, i].T, levels=[0.5], colors="orange")
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Flow in xy plane, t = {t:.1f}")
    plt.savefig(f"snapshots/velocity/xy_{n:04d}.png")
    plt.close()

    # xz plane
    plt.figure(figsize=(6, 6))
    plt.quiver(x[::skip], x[::skip], vx[::skip, i, ::skip].T, vz[::skip, i, ::skip].T)
    plt.contour(x, x, fU[:, i, :].T, levels=[0.5], colors="orange")
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("z")
    plt.title(f"Flow in xz plane, t = {t:.1f}")
    plt.savefig(f"snapshots/velocity/xz_{n:04d}.png")
    plt.close()

    # yz plane
    plt.figure(figsize=(6, 6))
    plt.quiver(x[::skip], x[::skip], vy[i, ::skip, ::skip].T, vz[i, ::skip, ::skip].T)
    plt.contour(x, x, fU[i, :, :].T, levels=[0.5], colors="orange")
    plt.gca().set_aspect("equal")
    plt.xlabel("y"); plt.ylabel("z")
    plt.title(f"Flow in yz plane, t = {t:.1f}")
    plt.savefig(f"snapshots/velocity/yz_{n:04d}.png")
    plt.close()

    print(f"Velocity plots for frame {n} saved")


# 4. Velocity field in the xy plane
#i = N // 2
#skip = 3 # skip every 3rd point for clarity
#plt.figure(figsize=(6, 6))
#plt.quiver(x[::skip], x[::skip], vx[::skip, ::skip, i].T, vy[::skip, ::skip, i].T)
#circle = plt.Circle((c, c), 4.5, fill=False, color="orange")
#plt.gca().add_patch(circle)
#plt.gca().set_aspect("equal")
#plt.xlabel("x"); plt.ylabel("y")
#plt.title("Flow in xy plane")
#plt.savefig("snapshots/04_velocity_slice_xy.png")

# 5. Velocity field in the xz plane
#i = N // 2
#skip = 3 # skip every 3rd point for clarity
#plt.figure(figsize=(6, 6))
#plt.quiver(x[::skip], x[::skip], vx[::skip, i, ::skip].T, vz[::skip, i, ::skip].T)
#circle = plt.Circle((c, c), 4.5, fill=False, color="orange")
#plt.gca().add_patch(circle)
#plt.gca().set_aspect("equal")
#plt.xlabel("x"); plt.ylabel("z")
#plt.title("Flow in xz plane")
#plt.savefig("snapshots/05_velocity_slice_xz.png")

# 6. Velocity field in the yz plane
#i = N // 2
#skip = 3 # skip every 3rd point for clarity
#plt.figure(figsize=(6, 6))
#plt.quiver(x[::skip], x[::skip], vy[i, ::skip, ::skip].T, vz[i, ::skip, ::skip].T)
#circle = plt.Circle((c, c), 4.5, fill=False, color="orange")
#plt.gca().add_patch(circle)
#plt.gca().set_aspect("equal")
#plt.xlabel("y"); plt.ylabel("z")
#plt.title("Flow in yz plane")
#plt.savefig("snapshots/06_velocity_slice_yz.png")

# 7. 3D flow: vortexringe (Q-kriterium), dråbe og streamlines
# frame = 12
# print("t =", times[frame])
# g = pv.ImageData(dimensions=vx.shape, spacing=(dx, dx, dx))
# g["v"] = np.column_stack([a.ravel(order="F") for a in (vx, vy, vz)])
# g["phi_U"] = frames_U[frame].ravel(order="F")
# g["Q"] = g.compute_derivative(scalars="v", qcriterion=True)["qcriterion"]
# seeds = pv.Line((c - 8, c, c), (c - 8, c + 18, c), resolution=12)
# p = pv.Plotter()
# p.add_mesh(g.contour([0.5], scalars="phi_U"), color="gold", opacity=0.3)
# p.add_mesh(g.contour([0.5 * g["Q"].max()], scalars="Q"), color="steelblue", opacity=0.5)
# p.add_mesh(g.streamlines_from_source(seeds, vectors="v").tube(radius=0.1), color="black")
# p.add_mesh(g.outline(), color="gray")
# p.add_axes()
# saving pyvista screenshot to file
# p.show(screenshot="snapshots/07_3D_flow.png")
# p.close()
