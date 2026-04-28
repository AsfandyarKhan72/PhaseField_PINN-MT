"""
Inverse PINN for 2D martensitic transformation using DeepXDE (PyTorch backend)

This script runs a batch of inverse-PINN experiments to infer the gradient-energy coefficient β
from synthetic observations of the order parameters η1, η2.

"""

# ============================================================
# BACKEND + IMPORTS
# ============================================================
import os
os.environ["DDE_BACKEND"] = "pytorch"

import json
from pathlib import Path

import numpy as np
import torch # torch==2.2.2+cu121
import deepxde as dde # deepxde==1.13.2
from deepxde.icbc import PointSetBC, PeriodicBC


# ============================================================
# GLOBAL SETTINGS
# ============================================================
SEED = 2026
dde.config.set_random_seed(SEED)
dde.config.set_default_float("float32")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()  # Clear residual GPU allocations from any previous runs


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(
    "/home/asfandyarkhan/Desktop/Papers/MT_Paper/CMAME/PINN_Code/Inverse/Final_Inverse_GitHub/"
    "Inverse_GitHub/New_Results/Final_Inverse_Results/Combined_Files"
)

RUNS_ROOT = BASE_DIR / "RUNS_BetaOnly"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PHYSICAL PARAMETERS (Tables 1-2 in manuscript)
# ============================================================
# Spatial domain: [0,1]x[0,1]
XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0

# Time domain (model time)
T0_MODEL, T1_MODEL = 0.0, 1.0

# Nondimensionalization scales (L0=30 μm, T0=1 ns)
# The computational domain is [0,1]^2 and time window [0,1].
L0 = 30e-6      # 30 μm
T0 = 1e-9       # 1 ns
len_sc = 1 / (L0**2)   # scaling factor for Laplacian under x = x_phys/L0
time_sc = T0           

# Thermodynamic / kinetic constants
DelF = 36e6  # chemical driving force ΔG (J mol^-1), Table 2
Lmob = 10.0   # kinetic coefficient L (m^3 J^-1 s^-1), Table 2

# Elastic constants (GPa)
C11, C12, C44 = 264.24, 115.38, 153.86

# Landau coefficients (dimensionless), Table 2
a, b, c = 0.2, -12.6, 12.4


# ============================================================
# OUTPUT TRANSFORM (SCALING)
# ============================================================
# These scaling factors stabilize multi-physics training.
u_scale = 1e-3
eta_scale = 1e-1

def output_transform(X, Y):
    """
    Scale raw NN outputs to improve conditioning:
        (u1,u2,eta1,eta2) -> (u1*u_scale, u2*u_scale, eta1*eta_scale, eta2*eta_scale)
    """
    u1_raw, u2_raw, eta1_raw, eta2_raw = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3], Y[:, 3:4]
    return torch.cat(
        [u1_raw * u_scale, u2_raw * u_scale, eta1_raw * eta_scale, eta2_raw * eta_scale],
        dim=1,
    )


# ============================================================
# GEOMETRY + TIME DOMAIN
# ============================================================
def make_geomtime():
    """
    Fixed geometry-time domain:

    """
    geom = dde.geometry.Rectangle(xmin=[XMIN, YMIN], xmax=[XMAX, YMAX])
    timedomain = dde.geometry.TimeDomain(T0_MODEL, T1_MODEL)
    return dde.geometry.GeometryXTime(geom, timedomain)


# ============================================================
# PERIODIC BCS FOR η1, η2
# ============================================================
def boundary_x(X, on_boundary):
    return on_boundary and (np.isclose(X[0], XMIN) or np.isclose(X[0], XMAX))

def boundary_y(X, on_boundary):
    return on_boundary and (np.isclose(X[1], YMIN) or np.isclose(X[1], YMAX))

def make_periodic_eta_bcs(geomtime):
    """
    Periodic BCs for eta fields:
      η(x=0,y,t)=η(x=1,y,t)
      η(x,y=0,t)=η(x,y=1,t)
    """
    bc_eta1_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=2)
    bc_eta2_x = PeriodicBC(geomtime, component_x=0, on_boundary=boundary_x, derivative_order=0, component=3)
    bc_eta1_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=2)
    bc_eta2_y = PeriodicBC(geomtime, component_x=1, on_boundary=boundary_y, derivative_order=0, component=3)
    return [bc_eta1_x, bc_eta2_x, bc_eta1_y, bc_eta2_y]


# ============================================================
# PDE FACTORY (β is trainable)
# ============================================================
def make_pde(beta_var: dde.Variable):
    """
    Returns PDE residual function using a *trainable* scalar β.
    PDE system (strong form):
      eq1, eq2 : mechanical equilibrium (∇·σ = 0)
      eq3, eq4 : TDGL evolution for η1, η2 with gradient energy β∇²η terms
    """
    def pde(x, y):
        # Fields: [u1, u2, eta1, eta2]
        u1, u2, eta1, eta2 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

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

        # Time derivatives
        eta1_t = dde.grad.jacobian(y, x, i=2, j=2)
        eta2_t = dde.grad.jacobian(y, x, i=3, j=2)

        # Stress tensor components (linear elasticity + transformation coupling)
        sigma11 = 7.443 * (eta2 ** 2 - eta1 ** 2) + C11 * u1_x + C12 * u2_y
        sigma12 = C44 * (u2_x + u1_y)
        sigma22 = 7.443 * (eta1 ** 2 - eta2 ** 2) + C12 * u1_x + C11 * u2_y

        # Mechanical equilibrium residuals
        eq1 = dde.grad.jacobian(sigma11, x, j=0) + dde.grad.jacobian(sigma12, x, j=1)
        eq2 = dde.grad.jacobian(sigma12, x, j=0) + dde.grad.jacobian(sigma22, x, j=1)

        # Gradient energy terms (β is trainable!)
        div_eta1 = -(Lmob * beta_var * time_sc * len_sc) * (eta1_xx + eta1_yy)
        div_eta2 = -(Lmob * beta_var * time_sc * len_sc) * (eta2_xx + eta2_yy)

        # Landau + elastic coupling driving forces 
        f1 = Lmob * (
            (1.4886e9 * time_sc * eta1 * (-eta1**2 + eta2**2 + 10*u1_x - 10*u2_y))
            - (DelF * time_sc * (a*eta1 + b*eta1**2 + c*eta1*(eta1**2 + eta2**2)))
        )
        f2 = -Lmob * (
            (1.4886e9 * time_sc * eta2 * (-eta1**2 + eta2**2 + 10*u1_x - 10*u2_y))
            + (DelF * time_sc * (a*eta2 + b*eta2**2 + c*eta2*(eta1**2 + eta2**2)))
        )

        # TDGL residuals
        eq3 = eta1_t + div_eta1 - f1
        eq4 = eta2_t + div_eta2 - f2

        return [eq1, eq2, eq3, eq4]

    return pde


# ============================================================
# DATA IO HELPERS
# ============================================================
def load_grid_file(path: Path):
    """
    Load a file containing columns:
        x, y, eta1, eta2

    Returns:
        xy  : (N,2)
        e1  : (N,1)
        e2  : (N,1)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Try CSV (no header)
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = None

    # If CSV parse gave 1-column, try whitespace
    if arr is None or (arr.ndim == 1) or (arr.shape[1] < 4):
        try:
            arr = np.loadtxt(path)  # whitespace
        except Exception as e:
            raise ValueError(f"Could not parse file: {path}\n{e}")

    # If still not enough columns, try skipping one header row
    if arr.ndim == 2 and arr.shape[1] < 4:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            arr = np.loadtxt(path, skiprows=1)

    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{path} must have >=4 columns: x,y,eta1,eta2. Got shape {arr.shape}")

    xy = arr[:, 0:2].astype(np.float64)
    e1 = arr[:, 2:3].astype(np.float64)
    e2 = arr[:, 3:4].astype(np.float64)
    return xy, e1, e2


def make_pointset_constraints(data_file: Path, t_value: float, kind: str):
    """
    Create PointSetBC constraints at a fixed time:
      - If kind='ic'  : treated as initial constraint (t=t_ic)
      - If kind='obs' : treated as observation constraint (t=t_obs)

    Returns:
      xyt : (N,3)
      bc1 : PointSetBC for eta1
      bc2 : PointSetBC for eta2
    """
    xy, e1, e2 = load_grid_file(data_file)
    tcol = np.full((xy.shape[0], 1), float(t_value), dtype=np.float64)
    xyt = np.hstack((xy, tcol))
    bc1 = PointSetBC(xyt, e1, component=2)
    bc2 = PointSetBC(xyt, e2, component=3)

    if kind not in ("ic", "obs"):
        raise ValueError("kind must be 'ic' or 'obs'")
    return xyt, bc1, bc2


# ============================================================
# SINGLE RUN
# ============================================================
def run_one_case(case_id: str, ic_file: Path, obs_file: Path, t_ic: float, t_obs: float, cfg: dict):
    """
    Run one inverse-PINN experiment:
      - enforce IC (eta1, eta2) at t=t_ic using PointSetBC
      - enforce observations (eta1, eta2) at t=t_obs using PointSetBC
      - infer scalar β by minimizing PDE + BC + IC + OBS losses
    """
    out_dir = RUNS_ROOT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==================== {case_id} ====================")
    print(f"IC  : {ic_file.name} @ t={t_ic}")
    print(f"OBS : {obs_file.name} @ t={t_obs}")
    print(f"OUT : {out_dir}")

    # --- fresh trainable parameter per run ---
    beta_trainable = dde.Variable(cfg["beta_init"])
    pde = make_pde(beta_trainable)

    # --- geometry-time & BCs ---
    geomtime = make_geomtime()
    periodic_bcs = make_periodic_eta_bcs(geomtime)

    # --- IC / OBS constraints ---
    xyt_ic, ic_eta1, ic_eta2 = make_pointset_constraints(ic_file, t_value=t_ic, kind="ic")
    xyt_obs, obs_eta1, obs_eta2 = make_pointset_constraints(obs_file, t_value=t_obs, kind="obs")

    # Anchors keep these points in the training set (in addition to PointSetBC)
    anchors = np.vstack((xyt_ic, xyt_obs))

    # --- network ---
    net = dde.nn.FNN(cfg["layer_size"], cfg["activation"], cfg["initializer"])
    net.apply_output_transform(output_transform)

    # --- data object ---
    # Loss-term order here:
    #   [PDE 4] + [PeriodicBC 4] + [IC 2] + [OBS 2]
    data = dde.data.TimePDE(
        geomtime,
        pde,
        periodic_bcs + [ic_eta1, ic_eta2, obs_eta1, obs_eta2],
        num_domain=cfg["num_domain"],
        num_boundary=cfg["num_boundary"],
        train_distribution=cfg["train_dist"],
        anchors=anchors,
        num_test=cfg["num_test"],
    )

    model = dde.Model(data, net)

    # --- callbacks ---
    pde_resampler = dde.callbacks.PDEPointResampler(
        period=cfg["resample_period"], pde_points=True, bc_points=True
    )

    beta_path = out_dir / "beta_vs_iterations.txt"
    beta_tracker = dde.callbacks.VariableValue(
        beta_trainable, period=cfg["beta_track_period"], filename=str(beta_path), precision=6
    )

    # --- loss weights  ---
    loss_weights = cfg["loss_weights"]

    # --- training schedule ---
    for stage_idx, stage in enumerate(cfg["train_stages"], start=1):
        lr = stage["lr"]
        iters = stage["iters"]
        print(f"Stage {stage_idx}: Adam lr={lr:.1e}, iters={iters}")

        model.compile(
            "adam",
            lr=lr,
            loss="MSE",
            loss_weights=loss_weights,
            external_trainable_variables=[beta_trainable],
        )
        model.train(
            iterations=iters,
            display_every=cfg["display_every"],
            callbacks=[beta_tracker, pde_resampler],
        )

    # --- save model + config ---
    model.save(str(out_dir / "model"))
    print("Saved model + beta history:", out_dir)

    run_cfg = {
        "case_id": case_id,
        "ic_file": str(ic_file),
        "obs_file": str(obs_file),
        "t_ic": float(t_ic),
        "t_obs": float(t_obs),
        "beta_init": float(cfg["beta_init"]),
        "loss_weights": loss_weights,
        "train_stages": cfg["train_stages"],
        "num_domain": cfg["num_domain"],
        "num_boundary": cfg["num_boundary"],
        "num_test": cfg["num_test"],
        "train_distribution": cfg["train_dist"],
        "seed": SEED,
        "device": DEVICE,
        "notes": "Inverse PINN (beta only) - periodic BCs for eta; IC and OBS via PointSetBC",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))


# ============================================================
# MAIN (batch plan)
# ============================================================
def main():
    # --------------------------
    # Config
    # --------------------------
    CONFIG = dict(
        # NN architecture (same style as forwardPINN)
        layer_size=[3] + [128] * 4 + [4],     # inputs: (x,y,t) -> outputs: (u1,u2,eta1,eta2)
        activation="tanh",
        initializer="Glorot uniform",

        # Data sampling
        num_domain=20000,
        num_boundary=4000,
        num_test=50000,
        train_dist="Hammersley",

        # Trainable parameter initialization
        beta_init=1.0,

        # Loss weights: [PDE4] + [PeriodicBC4] + [IC2] + [OBS2] = 12
        loss_weights=[1, 1, 1, 1,  1, 1, 1, 1,  1000, 1000,  1000, 1000],

        # Training schedule
        train_stages=[
            {"lr": 1e-3, "iters": 30000},
            {"lr": 1e-4, "iters": 10000},
            {"lr": 1e-5, "iters": 10000},
        ],

        # Callback settings
        resample_period=1000,
        beta_track_period=1000,
        display_every=1000,
    )

    # ----------------------------------------------
    #  Studies to run (Observed data grid matches IC)
    # ----------------------------------------------

    # 1) GRID STUDY
    grids = ["101x101", "51x51", "11x11"]
    for g in grids:
        obs = BASE_DIR / f"grid_{g}.txt"
        ic  = BASE_DIR / f"gridIC_{g}.txt"
        run_one_case(
            case_id=f"01_gridstudy_{g}_ICmatch",
            ic_file=ic,
            obs_file=obs,
            t_ic=0.0,
            t_obs=1.0,
            cfg=CONFIG,
        )

    # 2) NOISE STUDY (IC noise matches OBS noise)
    noise_levels = ["10pct", "20pct", "50pct"]
    for nl in noise_levels:
        ic  = BASE_DIR / f"gridIC_11x11_noise_{nl}.txt"
        obs = BASE_DIR / f"grid_11x11_noise_{nl}.txt"
        run_one_case(
            case_id=f"02_noisestudy_11x11_{nl}_ICnoise",
            ic_file=ic,
            obs_file=obs,
            t_ic=0.0,
            t_obs=1.0,
            cfg=CONFIG,
        )

    # 3) HALF-DOMAIN STUDY
    ic11 = BASE_DIR / "gridIC_11x11.txt"
    run_one_case(
        case_id="05_halfdata_firsthalf_11x11",
        ic_file=ic11,
        obs_file=BASE_DIR / "grid_11x11_firsthalf.txt",
        t_ic=0.0,
        t_obs=1.0,
        cfg=CONFIG,
    )
    run_one_case(
        case_id="06_halfdata_secondhalf_11x11",
        ic_file=ic11,
        obs_file=BASE_DIR / "grid_11x11_secondhalf.txt",
        t_ic=0.0,
        t_obs=1.0,
        cfg=CONFIG,
    )

    print("\n ALL inverse-beta runs completed.")
    print("Results root:", RUNS_ROOT)


if __name__ == "__main__":
    main()
