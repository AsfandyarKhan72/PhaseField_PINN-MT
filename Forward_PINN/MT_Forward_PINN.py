"""
Forward PINN for 2D martensitic transformation using DeepXDE (PyTorch backend)

Paper:
  "Physics-Informed Neural Networks for Martensitic Transformation:
   Toward Morphology-Based Parameter Estimation"

"""

# ============================================================
# BACKEND + IMPORTS
# ============================================================
import os
os.environ["DDE_BACKEND"] = "pytorch" 

from pathlib import Path
import json
import random
import numpy as np
import torch # torch==2.2.2+cu121
import deepxde as dde # deepxde==1.13.2
from deepxde.icbc import PeriodicBC, PointSetBC
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL SETTINGS (REPRODUCIBILITY)
# ============================================================

SEED = 2026
dde.config.set_random_seed(SEED)
dde.config.set_default_float("float32")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================

# Repository root = parent of the current script folder
REPO_ROOT = Path(__file__).resolve().parents[1]

# Data folder inside repository
BASE_DIR = REPO_ROOT / "Forward_PINN"

# FEM reference file
FEM_END_FILE = BASE_DIR / "t_1_FEM.txt"

# Output folder inside repository
OUT_ROOT = REPO_ROOT / "results" / "forward"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# PHYSICAL PARAMETERS (Tables 1-2 in manuscript)
# ============================================================

# Elastic constants (GPa), used in strong-form mechanical equilibrium residuals
C11 = 264.24
C12 = 115.38
C44 = 153.86

# TDGL / chemical parameters
Beta = 1e-4     # gradient energy coefficient (J m^-1), Table 2
DelF = 36e6     # chemical driving force ΔG (J mol^-1), Table 2
Lmob = 10.0     # kinetic coefficient L (m^3 J^-1 s^-1), Table 2

# Landau coefficients (dimensionless), Table 2
a, b, c = 0.2, -12.6, 12.4

# Nondimensionalization scales (L0=30 μm, T0=1 ns)
# The computational domain is [0,1]^2 and time window [0,1].
L0 = 30e-6      # 30 μm
T0 = 1e-9       # 1 ns
len_sc = 1 / (L0**2)   # scaling factor for Laplacian under x = x_phys/L0
time_sc = T0           


# ============================================================
# COMPUTATIONAL DOMAIN
# ============================================================

XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0
T0_MODEL, T1_MODEL = 0.0, 1.0


# ============================================================
# NETWORK OUTPUT SCALING (stabilizes multi-physics optimization)
# ============================================================

# Displacements are typically small compared to order parameters; scaling improves conditioning.
u_scale = 1e-5
eta_scale = 1e-1

def output_transform(X, Y):
    """
    Apply a simple output scaling to balance magnitudes across fields:
      (u1,u2,eta1,eta2) -> (u1*u_scale, u2*u_scale, eta1*eta_scale, eta2*eta_scale)
    """
    u1, u2, eta1, eta2 = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4]
    return torch.cat([u1*u_scale, u2*u_scale, eta1*eta_scale, eta2*eta_scale], dim=1)


# ============================================================
# GOVERNING EQUATIONS: MECHANICS + TDGL (strong form residuals)
# ============================================================

def pde(x, y):
    """
    Governing equations:
    Coupled PDE system (4 residuals):
      eq1, eq2: mechanical equilibrium: ∂σ11/∂x + ∂σ12/∂y = 0,  ∂σ12/∂x + ∂σ22/∂y = 0
      eq3, eq4: TDGL for order parameters η1 and η2 (time-dependent): ∂η_p/∂t = -L( -β∇²η_p + ∂f/∂η_p + δF_el/δη_p ), p=1,2

    """
    u1, u2, eta1, eta2 = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4]

    # Displacement gradients
    u1_x = dde.grad.jacobian(y, x, i=0, j=0)
    u1_y = dde.grad.jacobian(y, x, i=0, j=1)
    u2_x = dde.grad.jacobian(y, x, i=1, j=0)
    u2_y = dde.grad.jacobian(y, x, i=1, j=1)

    # Laplacians of order parameters
    eta1_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    eta1_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
    eta2_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    eta2_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)

    # Time derivatives of order parameters
    eta1_t = dde.grad.jacobian(y, x, i=2, j=2)
    eta2_t = dde.grad.jacobian(y, x, i=3, j=2)

    # --- Stress components (linear elasticity + transformation strain coupling) ---
    # (Transformation strain tensors and material constants are described in Table 1.)
    sigma11 = 7.443*(eta2**2 - eta1**2) + C11*u1_x + C12*u2_y
    sigma12 = C44*(u2_x + u1_y)
    sigma22 = 7.443*(eta1**2 - eta2**2) + C12*u1_x + C11*u2_y

    # Mechanical equilibrium residuals (Eq. 9)
    eq1 = dde.grad.jacobian(sigma11, x, j=0) + dde.grad.jacobian(sigma12, x, j=1)
    eq2 = dde.grad.jacobian(sigma12, x, j=0) + dde.grad.jacobian(sigma22, x, j=1)

    # Gradient energy terms (scaled Laplacian)
    div1 = -(Lmob * Beta * time_sc * len_sc) * (eta1_xx + eta1_yy)
    div2 = -(Lmob * Beta * time_sc * len_sc) * (eta2_xx + eta2_yy)

    # Landau driving force + elastic coupling term
    f1 = Lmob * (
        (1.4886e9 * time_sc * eta1 * (-eta1**2 + eta2**2 + 10*u1_x - 10*u2_y))
        - (DelF * time_sc * (a*eta1 + b*eta1**2 + c*eta1*(eta1**2 + eta2**2)))
    )
    f2 = -Lmob * (
        (1.4886e9 * time_sc * eta2 * (-eta1**2 + eta2**2 + 10*u1_x - 10*u2_y))
        + (DelF * time_sc * (a*eta2 + b*eta2**2 + c*eta2*(eta1**2 + eta2**2)))
    )

    # TDGL residuals (Eq. 8)
    eq3 = eta1_t + div1 - f1
    eq4 = eta2_t + div2 - f2

    return [eq1, eq2, eq3, eq4]


# ============================================================
# PERIODIC BCs ON η1, η2
# ============================================================
"""
Boundary conditions:
  - Periodic BCs on η1 and η2 along x=0/1 and y=0/1.
  - No explicit displacement BCs (u1,u2 unconstrained) because rigid-body translation does not affect
    the stress/strain-driven phase evolution in the strong-form PINN setting.
"""

def boundary_x(X, on_boundary):
    return on_boundary and (np.isclose(X[0], XMIN) or np.isclose(X[0], XMAX))

def boundary_y(X, on_boundary):
    return on_boundary and (np.isclose(X[1], YMIN) or np.isclose(X[1], YMAX))

def make_periodic_bcs(geomtime):
    """
    Periodic BCs for η1 and η2:
      η(x=0,y,t)=η(x=1,y,t) and η(x,y=0,t)=η(x,y=1,t)
    """
    bc_eta1_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=2)
    bc_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=3)
    bc_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=2)
    bc_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=3)
    return [bc_eta1_x, bc_eta2_x, bc_eta1_y, bc_eta2_y]


# ============================================================
# INITIAL CONDITIONS: Gaussian perturbation centered at (0.3, 0.3)
# (Eq. 10 in manuscript)
# ============================================================
"""
Initial condition:
  - Manufactured Gaussian perturbations (nucleation seed) centered at (0.3, 0.3):
      η_p(x,y,0) = A_p + B_p exp( -((x-0.3)² + (y-0.3)²)/(2σ²) ),  p=1,2
    with (A1,B1)=(0.1,0.4), (A2,B2)=(0.5,-0.4), σ=0.05
"""

# ============================================================
# INITIAL CONDITIONS (analytic IC enforced via dde.IC)
# ============================================================

SIGMA = 0.05
IC_CENTER_X = 0.3
IC_CENTER_Y = 0.3

def eta1_ic_func(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.1 + 0.4 * np.exp(-((x - IC_CENTER_X)**2 + (y - IC_CENTER_Y)**2) / (2 * SIGMA**2))

def eta2_ic_func(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.5 - 0.4 * np.exp(-((x - IC_CENTER_X)**2 + (y - IC_CENTER_Y)**2) / (2 * SIGMA**2))

def initial_time(X, on_initial):
    # DeepXDE passes X = [x, y, t]
    return on_initial and np.isclose(X[2], T0_MODEL)  # usually 0.0

def make_analytic_ics(geomtime):
    ic_eta1 = dde.IC(geomtime, eta1_ic_func, initial_time, component=2)
    ic_eta2 = dde.IC(geomtime, eta2_ic_func, initial_time, component=3)
    return [ic_eta1, ic_eta2]



# ============================================================
# FEM LOADER + IC SAVING
# ============================================================
def load_fem_file(path: Path):
    """
    Load FEM reference microstructure data.
    Expected FEM file format:
        x, y, eta1, eta2

    """
    if not path.exists():
        raise FileNotFoundError(f"Missing FEM file: {path}")

    try:
        # Trying comma-separated first
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        # Fallback: whitespace-separated without header
        arr = np.loadtxt(path)

    if arr.shape[1] < 4:
        raise ValueError(
            f"FEM file must contain at least 4 columns: x, y, eta1, eta2. "
            f"Got array shape {arr.shape}"
        )

    xy = arr[:, :2].astype(np.float64)
    e1 = arr[:, 2].astype(np.float64)
    e2 = arr[:, 3].astype(np.float64)

    # Sort points to preserve structured grid ordering (important for contour plots)
    idx = np.lexsort((xy[:, 1], xy[:, 0]))

    return xy[idx], e1[idx], e2[idx]

def save_ic_from_prediction(out_path: Path, xy, eta1, eta2, delimiter=","):
    """
    Save a PINN-predicted microstructure snapshot as an IC file for the next window.
    File format (comma-separated):
        x, y, eta1, eta2

    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([xy[:, 0], xy[:, 1], eta1, eta2]).astype(np.float64)
    np.savetxt(out_path, arr, delimiter=delimiter, header="x,y,eta1,eta2", comments="")
    return out_path


# ============================================================
# ERROR METRICS + eta EVALUATION
# ============================================================
def relative_L2(pred, ref, eps=1e-14):
    """
    Compute the relative L2 error norm between PINN prediction and FEM reference.
    Definition:
        Relative L2 = || pred - ref ||_2 / || ref ||_2
    """

    num = np.sqrt(np.sum((pred - ref) ** 2))
    den = np.sqrt(np.sum(ref ** 2))
    return float(num / max(den, eps))

def linf(pred, ref):
    """
    Compute the Linf (maximum absolute) error norm.
    Definition:
        Linf = max | pred - ref |
    """
    return float(np.max(np.abs(pred - ref)))

def predict_eta(model, xy, t):
    """
    Evaluate trained PINN model for eta1 and eta2 on a given spatial grid and time.
    Note:
        The PINN outputs four fields:
            [u1, u2, eta1, eta2]
        Here we extract only the phase-field variables eta1 and eta2,
        which are used for comparison with FEM reference microstructures.
    """
    X = np.hstack([xy.astype(np.float64), np.full((xy.shape[0], 1), float(t), dtype=np.float64)])
    Y = model.predict(X)
    return Y[:, 2].astype(np.float64), Y[:, 3].astype(np.float64)


# ============================================================
# VISUALIZATION UTILITIES
# ============================================================
def save_contour_comparison(out_dir: Path, tag: str, xy, ref1, ref2, p1, p2):
    """
    Contour visualization for structured grids.
    Produces:
      Ref η1, PINN η1, |err| η1
      Ref η2, PINN η2, |err| η2
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gx = np.unique(xy[:, 0])
    gy = np.unique(xy[:, 1])
    nx, ny = len(gx), len(gy)

    def reshape(z):
        return z.reshape(ny, nx)

    R1, R2 = reshape(ref1), reshape(ref2)
    P1, P2 = reshape(p1), reshape(p2)
    E1, E2 = np.abs(P1 - R1), np.abs(P2 - R2)

    vmin1, vmax1 = min(R1.min(), P1.min()), max(R1.max(), P1.max())
    vmin2, vmax2 = min(R2.min(), P2.min()), max(R2.max(), P2.max())

    levels = 200
    cmap = "jet"

    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(tag, fontsize=11)

    im = ax[0,0].contourf(gx, gy, R1, levels=levels, cmap=cmap, vmin=vmin1, vmax=vmax1); ax[0,0].set_title("Ref η1"); ax[0,0].axis("off"); fig.colorbar(im, ax=ax[0,0])
    im = ax[0,1].contourf(gx, gy, P1, levels=levels, cmap=cmap, vmin=vmin1, vmax=vmax1); ax[0,1].set_title("PINN η1"); ax[0,1].axis("off"); fig.colorbar(im, ax=ax[0,1])
    im = ax[0,2].contourf(gx, gy, E1, levels=levels, cmap=cmap); ax[0,2].set_title("|err| η1"); ax[0,2].axis("off"); fig.colorbar(im, ax=ax[0,2])

    im = ax[1,0].contourf(gx, gy, R2, levels=levels, cmap=cmap, vmin=vmin2, vmax=vmax2); ax[1,0].set_title("Ref η2"); ax[1,0].axis("off"); fig.colorbar(im, ax=ax[1,0])
    im = ax[1,1].contourf(gx, gy, P2, levels=levels, cmap=cmap, vmin=vmin2, vmax=vmax2); ax[1,1].set_title("PINN η2"); ax[1,1].axis("off"); fig.colorbar(im, ax=ax[1,1])
    im = ax[1,2].contourf(gx, gy, E2, levels=levels, cmap=cmap); ax[1,2].set_title("|err| η2"); ax[1,2].axis("off"); fig.colorbar(im, ax=ax[1,2])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"{tag}.png", dpi=300)
    plt.close(fig)


# ============================================================
# ADAPTIVE REFINEMENT
# ============================================================

def _unique_rows_tol(X, tol=1e-12):
    Xr = np.round(X / tol) * tol
    _, idx = np.unique(Xr, axis=0, return_index=True)
    return np.sort(idx)

def select_top_residual_points(model, geomtime, n_each=100, pool=80000, tol=1e-12):
    """
    Adaptive sampling driven by TDGL residuals:
      - Evaluate residuals on a large candidate pool in space-time.
      - Rank points by |eq3| (η1 TDGL) and |eq4| (η2 TDGL).
      - Select top n_each from each residual and merge (unique points).
    """
    X_pool = geomtime.random_points(pool)
    res = model.predict(X_pool, operator=pde)  # list: [eq1, eq2, eq3, eq4]

    r3 = np.abs(res[2]).reshape(-1)
    r4 = np.abs(res[3]).reshape(-1)

    idx3 = np.argsort(-r3)[:n_each]
    idx4 = np.argsort(-r4)[:n_each]

    X_sel = np.vstack([X_pool[idx3], X_pool[idx4]])
    uidx = _unique_rows_tol(X_sel, tol=tol)
    X_unique = X_sel[uidx]

    # log details (kind + score)
    info_rows = []
    for ii in idx3:
        info_rows.append([X_pool[ii,0], X_pool[ii,1], X_pool[ii,2], "abs_eq3", float(r3[ii])])
    for ii in idx4:
        info_rows.append([X_pool[ii,0], X_pool[ii,1], X_pool[ii,2], "abs_eq4", float(r4[ii])])

    return X_unique, np.array(info_rows, dtype=object)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(cfg: dict):
    geom = dde.geometry.Rectangle([XMIN, YMIN], [XMAX, YMAX])
    timedomain = dde.geometry.TimeDomain(T0_MODEL, T1_MODEL)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Periodic BCs for order parameters
    bcs = make_periodic_bcs(geomtime)

    # Analytic ICs (old style)
    ics = make_analytic_ics(geomtime)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        bcs + ics,
        num_domain=cfg["num_domain"],
        num_boundary=cfg["num_boundary"],
        num_initial=cfg["num_initial"],      
        train_distribution=cfg["train_dist"],
        num_test=cfg["num_test"],
    )

    net = dde.nn.FNN(cfg["layer_size"], cfg["activation"], cfg["initializer"])
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    return model, geomtime



# ============================================================
# TRAINING ROUTINES
# ============================================================
def train_adam_lbfgs(model, out_dir: Path, cfg: dict, tag: str):
    """
    Two-stage optimization:
      1) Adam (stochastic) to reach a reasonable basin
      2) L-BFGS (quasi-Newton) for final convergence
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ADAM
    model.compile("adam", lr=cfg["adam_lr"], loss="MSE", loss_weights=cfg["loss_weights"])
    model.train(
        iterations=cfg["adam_iters"],
        display_every=cfg["display_every"],
        model_save_path=str(out_dir / f"{tag}_adam")
    )

    # L-BFGS
    dde.optimizers.config.set_LBFGS_options(
        maxcor=100,
        ftol=1e-12,
        gtol=1e-10,
        maxiter=cfg["lbfgs_iters"],
        maxfun=None,
        maxls=50
    )
    model.compile("L-BFGS", loss="MSE", loss_weights=cfg["loss_weights"])
    model.train(model_save_path=str(out_dir / f"{tag}_lbfgs"))

    return model


# ============================================================
# EVALUATION: errors + plots at t=0 and t=1
# ============================================================
def make_grid_xy(n=101):
    xs = np.linspace(0.0, 1.0, n, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    return xy

def evaluate_and_report(model, out_dir: Path):
    """
    - t=0: compare PINN(t=0) vs IC pointset (scatter parity)
    - t=1: compare PINN(t=1) vs FEM reference (relative L2, Linf + contour plot)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- t=0 contour diagnostics on a structured grid ---
    xy0 = make_grid_xy(n=101)
    X0 = np.hstack([xy0, np.zeros((xy0.shape[0], 1), dtype=np.float64)])
    ic1_0 = eta1_ic_func(X0).reshape(-1)
    ic2_0 = eta2_ic_func(X0).reshape(-1)
    p1_0g, p2_0g = predict_eta(model, xy0, t=0.0)

    save_contour_comparison(out_dir, "t0_IC_vs_PINN_contours", xy0, ic1_0, ic2_0, p1_0g, p2_0g)

    # --- t=1 comparison against FEM ---
    xy_ref, e1_ref, e2_ref = load_fem_file(FEM_END_FILE)
    p1_1, p2_1 = predict_eta(model, xy_ref, t=1.0)

    # Save end-state to reuse as IC in the next window
    ic_outfile = out_dir / "IC_for_nextWindow_fromPINN_t1.txt"
    save_ic_from_prediction(ic_outfile, xy_ref, p1_1, p2_1, delimiter=",")

    L2_1 = relative_L2(p1_1, e1_ref)
    L2_2 = relative_L2(p2_1, e2_ref)
    Li1  = linf(p1_1, e1_ref)
    Li2  = linf(p2_1, e2_ref)

    # Save metrics
    np.savetxt(
        out_dir / "errors_t1.csv",
        np.array([[L2_1, L2_2, Li1, Li2]]),
        delimiter=",",
        header="relL2_eta1,relL2_eta2,Linf_eta1,Linf_eta2",
        comments=""
    )

    # Contour comparison at t=1
    save_contour_comparison(out_dir, "t1_FEM_vs_PINN_contours", xy_ref, e1_ref, e2_ref, p1_1, p2_1)

    # Print metrics to console
    print("\n--- FINAL METRICS at t = 1 ---")
    print(f"Relative L2 (eta1): {L2_1:.6e}")
    print(f"Relative L2 (eta2): {L2_2:.6e}")
    print(f"Linf        (eta1): {Li1:.6e}")
    print(f"Linf        (eta2): {Li2:.6e}")

    return dict(relL2_eta1=L2_1, relL2_eta2=L2_2, Linf_eta1=Li1, Linf_eta2=Li2)


# ============================================================
# MAIN DRIVER: baseline + optional adaptive refinement rounds
# ============================================================
def save_cfg(out_dir: Path, cfg: dict, filename: str = "cfg.json"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(cfg, indent=2))

def main():
    # --------------------------
    # Default configuration used in forward PINN runs
    # --------------------------
    CFG = dict(
        # Network architecture (paper: 4 hidden layers × 128 neurons, tanh)
        layer_size=[3] + [128]*4 + [4],  # inputs: (x,y,t), outputs: (u1,u2,eta1,eta2)
        activation="tanh",
        initializer="Glorot uniform",

        # Collocation/boundary/test points
        num_domain=30000,
        num_boundary=4000,
        num_initial=512,   
        num_test=50000,
        train_dist="Hammersley",

        # Optimization schedule
        adam_lr=1e-3,
        adam_iters=50000,
        lbfgs_iters=50000,
        display_every=1000,

        # Loss weights (stronger weight on TDGL / IC terms).
        # Order: [eq1,eq2,eq3,eq4,bc1,bc2,bc3,bc4,ic_eta1,ic_eta2]
        loss_weights=[1, 1, 100, 100, 1, 1, 1, 1, 100, 100],

        # Adaptive refinement controls
        use_adaptive_refinement=True,   # set False to run baseline only
        refine_rounds=3,
        add_each_eq=100,                # top-100 for η1 residual and top-100 for η2 residual
        pool=80000,                     # candidate pool size for ranking residuals
        refine_adam_iters=30000,
        refine_lbfgs_iters=30000,
    )

    # --------------------------
    # Build model for t ∈ [0,1]
    # --------------------------
    model, geomtime = build_model(CFG)

    # --------------------------
    # Baseline training (no refinement)
    # --------------------------
    base_dir = OUT_ROOT / "baseline_no_refinement"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_cfg(base_dir, CFG)


    print("\n==================== BASELINE TRAINING ====================")
    model = train_adam_lbfgs(model, base_dir, CFG, tag="baseline")

    # Evaluate baseline
    baseline_metrics = evaluate_and_report(model, base_dir)
    (base_dir / "summary.json").write_text(json.dumps({"baseline": baseline_metrics}, indent=2))

    # --------------------------
    # Adaptive refinement rounds (optional)
    # --------------------------
    if not CFG["use_adaptive_refinement"]:
        print("\nAdaptive refinement disabled. Done.")
        return

    print("\n==================== ADAPTIVE REFINEMENT ====================")
    summary = {"baseline": baseline_metrics, "rounds": []}

    for rr in range(1, CFG["refine_rounds"] + 1):
        rr_dir = OUT_ROOT / f"adaptive_round_{rr:02d}"
        rr_dir.mkdir(parents=True, exist_ok=True)

        save_cfg(rr_dir, CFG)

        # 1) Select residual-driven anchors (top points from eq3 & eq4)
        X_new, info = select_top_residual_points(
            model=model,
            geomtime=geomtime,
            n_each=CFG["add_each_eq"],
            pool=CFG["pool"],
            tol=1e-12
        )

        # 2) Add anchors to the training set
        model.data.add_anchors(X_new)

        # Save anchors and ranking info
        np.savetxt(rr_dir / "anchors_added_X.csv", X_new, delimiter=",", header="x,y,t", comments="")
        with open(rr_dir / "anchors_added_info.csv", "w") as f:
            f.write("x,y,t,kind,score\n")
            for row in info:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

        # 3) Retrain (short schedule per round)
        print(f"\n--- Refinement round {rr}/{CFG['refine_rounds']} ---")
        model.compile("adam", lr=CFG["adam_lr"], loss="MSE", loss_weights=CFG["loss_weights"])
        model.train(
            iterations=CFG["refine_adam_iters"],
            display_every=CFG["display_every"],
            model_save_path=str(rr_dir / f"round{rr:02d}_adam")
        )

        dde.optimizers.config.set_LBFGS_options(
            maxcor=100, ftol=1e-12, gtol=1e-10,
            maxiter=CFG["refine_lbfgs_iters"], maxfun=None, maxls=50
        )
        model.compile("L-BFGS", loss="MSE", loss_weights=CFG["loss_weights"])
        model.train(model_save_path=str(rr_dir / f"round{rr:02d}_lbfgs"))

        # 4) Evaluate after refinement
        met = evaluate_and_report(model, rr_dir)

        summary["rounds"].append({
            "round": rr,
            "n_added": int(X_new.shape[0]),
            "metrics_t1": met,
        })
        (OUT_ROOT / "summary_all.json").write_text(json.dumps(summary, indent=2))

    print("\n All results saved in:")
    print("  ", OUT_ROOT)


if __name__ == "__main__":
    main()
