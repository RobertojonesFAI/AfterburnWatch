"""DF Prediction -- USGS M1 likelihood, recalibrated from actual observations.

Whiteboard: "DF Prediction". This is where the feedback loop closes:
satellite-identified debris flows (identify.py) become observations
(observations.py), and those observations pull the M1 coefficients toward
what actually happens on the ground.

Naming note: USGS "M1" (Staley et al. 2017) is the likelihood model below.
It is NOT the whiteboard's "Model 1", which is our satellite debris-flow
identifier in identify.py.

USGS M1 (Staley et al. 2017)
----------------------------
    x = B + Ct*T*R + Cf*F*R + Cs*S*R
    p = 1 / (1 + exp(-x))

    T  proportion of upslope area burned at moderate/high severity with
       slope >= 23 degrees
    F  mean dNBR / 1000 of the upslope area
    S  mean soil KF-factor of the upslope area
    R  peak rainfall *accumulation* (mm) over the design duration
       (15, 30 or 60 min)

In our Project Framework, T, F and S per basin/segment come from
wildcat/pfdf runs on Lemhi HPC (with RECOVER dNBR and soils as inputs). R
comes from the design storm, or for a real event from rainfall records
(MRMS -- the Debris Flow Hunters team's area).

Recalibration
-------------
``recalibrate`` fits B, Ct, Cf, Cs by maximum a posteriori logistic
regression with a Gaussian prior centered on the prior coefficients
(the loop always passes the published Staley 2017 values):

    minimize  -loglik(theta)  +  (prior_strength / 2) * ||theta - theta_prior||^2

With little new data the coefficients stay close to the published USGS
values. As trusted observations accumulate, they move toward what the new
data supports. ``prior_strength`` sets how much evidence it takes to move
them.

Keith Weber flagged that the thresholds were trained mostly on California
and that Idaho soils differ. Region-specific recalibration is the direct
answer to "would region-specific models be better?".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# Staley et al. (2017) M1 coefficients by rainfall duration (minutes):
# (B, Ct, Cf, Cs). Cross-check against pfdf before relying on them:
#   from pfdf.models.staley2017 import M1; M1.parameters()
STALEY2017_M1 = {
    15: (-3.63, 0.41, 0.67, 0.70),
    30: (-3.61, 0.26, 0.39, 0.50),
    60: (-3.21, 0.17, 0.20, 0.22),
}


@dataclass
class M1Coefficients:
    B: float
    Ct: float
    Cf: float
    Cs: float
    duration_min: int = 15
    source: str = "staley2017"
    n_obs: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def staley2017(cls, duration_min: int = 15) -> "M1Coefficients":
        if duration_min not in STALEY2017_M1:
            raise ValueError(f"duration must be one of {sorted(STALEY2017_M1)}")
        B, Ct, Cf, Cs = STALEY2017_M1[duration_min]
        return cls(B, Ct, Cf, Cs, duration_min=duration_min)

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.B, self.Ct, self.Cf, self.Cs], dtype="float64")

    def to_dict(self) -> dict:
        return asdict(self)


def _design(T, F, S, R) -> np.ndarray:
    T, F, S, R = (np.asarray(v, dtype="float64") for v in (T, F, S, R))
    return np.column_stack([np.ones_like(T), T * R, F * R, S * R])


def likelihood(coef: M1Coefficients, T, F, S, R) -> np.ndarray:
    """Debris-flow likelihood p (0-1) for each basin."""
    return expit(_design(T, F, S, R) @ coef.theta)


def rainfall_threshold(coef: M1Coefficients, T, F, S, p: float = 0.5) -> np.ndarray:
    """Rainfall accumulation R (mm) at which likelihood reaches ``p``.

    Inverts M1: R = (logit(p) - B) / (Ct*T + Cf*F + Cs*S). Returns NaN
    where the denominator is <= 0 (the model never reaches ``p``).
    """
    T, F, S = (np.asarray(v, dtype="float64") for v in (T, F, S))
    denom = coef.Ct * T + coef.Cf * F + coef.Cs * S
    logit = np.log(p / (1 - p))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (logit - coef.B) / denom, np.nan)
    return r


def _usable(obs: pd.DataFrame) -> pd.DataFrame:
    return obs.dropna(subset=["T", "F", "S", "R", "response"])


def recalibrate(
    prior: M1Coefficients,
    obs: pd.DataFrame,
    prior_strength: float = 2.0,
    min_obs: int = 30,
) -> tuple[M1Coefficients, dict]:
    """Update M1 coefficients from observations (MAP with Gaussian prior).

    ``obs`` needs columns T, F, S, R, response. Pass only *trusted*
    observations (observations.trusted). Returns ``(new_coefficients,
    info)``. With fewer than ``min_obs`` usable rows, or with only one
    response class, the prior comes back unchanged and ``info["reason"]``
    says why.
    """
    data = _usable(obs)
    info: dict = {"n_usable": int(len(data)), "prior": prior.to_dict()}
    if len(data) < min_obs:
        info["updated"] = False
        info["reason"] = f"only {len(data)} usable observations (< {min_obs})"
        return prior, info
    y = data["response"].to_numpy(dtype="float64")
    if y.min() == y.max():
        info["updated"] = False
        info["reason"] = "observations contain only one response class"
        return prior, info

    X = _design(data["T"], data["F"], data["S"], data["R"])
    theta0 = prior.theta
    lam = float(prior_strength)

    def objective(theta):
        z = X @ theta
        # log(1 + e^z) computed stably
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        diff = theta - theta0
        grad = X.T @ (expit(z) - y) + lam * diff
        return nll + 0.5 * lam * diff @ diff, grad

    res = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    if not res.success:
        info["updated"] = False
        info["reason"] = f"optimizer did not converge: {res.message}"
        return prior, info

    B, Ct, Cf, Cs = (float(v) for v in res.x)
    new = M1Coefficients(
        B,
        Ct,
        Cf,
        Cs,
        duration_min=prior.duration_min,
        source=f"recalibrated_from:{prior.source}",
        n_obs=int(len(data)),
    )
    info.update(
        updated=True,
        new=new.to_dict(),
        delta={k: float(v) for k, v in zip(["B", "Ct", "Cf", "Cs"], res.x - theta0)},
    )
    return new, info


def evaluate(coef: M1Coefficients, obs: pd.DataFrame) -> dict:
    """Skill of a coefficient set on observations: ROC AUC, Brier score, log loss."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    data = _usable(obs)
    if len(data) == 0:
        return {"n": 0}
    y = data["response"].to_numpy()
    p = np.clip(likelihood(coef, data["T"], data["F"], data["S"], data["R"]), 1e-9, 1 - 1e-9)
    out = {"n": int(len(data)), "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out
