"""SARS-CoV-2 RBD one-shot pre-registered test.

Pre-registration: invariants_preregistration_v4_rbd.md, locked in same commit.

Primary:  rho(PR_i, mean|Delta_bind_avg|_i) across ~194 RBD residues.
Direction predicted: NEGATIVE.
Four-tier decision rule with methodology-null gate.

No post-hoc rescue. One-shot. If fails, fails.
"""
import json, os, subprocess, sys, urllib.request
import numpy as np
import pandas as pd
from scipy import stats as sp
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'tables')
FIGS = os.path.join(ROOT, 'figures')
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

PRE_REG_COMMIT_FILE = os.path.join(HERE, 'invariants_preregistration_v4_rbd.md')

# LOCKED parameters
PDB_ID = '6M0J'
PDB_CHAIN = 'E'
LAMBDA_A = 8.0
MIN_MUTANTS = 10
RNG_SEED = 20260424
N_BOOT = 10_000
N_PERM = 1_000

# Secondary exploratory
LAMBDA_SENSITIVITY = [6.0, 10.0, 12.0]
RBM_RANGE = (437, 507)

# Data URLs
PDB_URL = f'https://files.rcsb.org/download/{PDB_ID}.pdb'
DMS_URLS = [
    # LFS-resolved URL (media subdomain serves the actual file, not the pointer)
    'https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/master/results/single_mut_effects/single_mut_effects.csv',
    'https://media.githubusercontent.com/media/jbloomlab/SARS-CoV-2-RBD_DMS/main/results/single_mut_effects/single_mut_effects.csv',
]


def get_pre_reg_commit():
    try:
        out = subprocess.run(
            ['git', 'log', '-n', '1', '--pretty=format:%H', '--', PRE_REG_COMMIT_FILE],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()[:10]
    except Exception:
        return 'unknown'


def download_pdb(cache_dir):
    path = os.path.join(cache_dir, f'{PDB_ID}.pdb')
    if not os.path.exists(path):
        print(f'[rbd] downloading {PDB_URL}...', flush=True)
        urllib.request.urlretrieve(PDB_URL, path)
    return path


def download_dms(cache_dir):
    path = os.path.join(cache_dir, 'single_mut_effects.csv')
    if os.path.exists(path):
        return path
    last_err = None
    for url in DMS_URLS:
        try:
            print(f'[rbd] downloading {url}', flush=True)
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f'failed all DMS URLs; last error: {last_err}')


def parse_pdb_chain(pdb_path, chain):
    """Return dict {residue_number: CA_coord (3,) numpy array} for the chain."""
    ca = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            if line[21:22] != chain: continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA': continue
            resnum = int(line[22:26].strip())
            altloc = line[16:17].strip()
            if altloc and altloc != 'A': continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            if resnum not in ca:
                ca[resnum] = np.array([x, y, z])
    return ca


def per_residue_PR(coords, lam):
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    c = np.exp(-d / lam)
    s1 = c.sum(axis=1)
    s2 = (c ** 2).sum(axis=1)
    n = len(coords)
    return (s1 ** 2) / (n * s2)


def spearman_ci(x, y, n_boot=N_BOOT, seed=RNG_SEED):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4: return None
    rho, p = sp.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ii = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(ii)) < 4:
            boots[i] = np.nan; continue
        boots[i] = sp.spearmanr(x[ii], y[ii]).statistic
    boots = boots[np.isfinite(boots)]
    return {'rho': float(rho), 'p': float(p), 'n': int(len(x)),
            'ci_lo': float(np.percentile(boots, 2.5)),
            'ci_hi': float(np.percentile(boots, 97.5))}


def main():
    commit = get_pre_reg_commit()
    print(f'[rbd] pre-reg commit: {commit}')

    cache = os.path.join(ROOT, 'data', 'rbd_cache')
    os.makedirs(cache, exist_ok=True)

    pdb_path = download_pdb(cache)
    dms_path = download_dms(cache)

    # -------- Parse PDB --------
    print(f'[rbd] parsing PDB {PDB_ID} chain {PDB_CHAIN}...', flush=True)
    ca = parse_pdb_chain(pdb_path, PDB_CHAIN)
    if len(ca) == 0:
        # Try alternate chains if E empty
        for alt in ['A', 'B', 'C', 'D', 'F']:
            ca = parse_pdb_chain(pdb_path, alt)
            if len(ca) > 100:
                print(f'  chain {PDB_CHAIN} empty, using chain {alt} instead ({len(ca)} residues)')
                break
    residue_ids = sorted(ca.keys())
    coords = np.array([ca[r] for r in residue_ids])
    print(f'  {len(residue_ids)} residues with CA, range {residue_ids[0]}-{residue_ids[-1]}')

    # -------- Compute PR --------
    pr_primary = per_residue_PR(coords, LAMBDA_A)
    print(f'  PR (lambda={LAMBDA_A}): mean={pr_primary.mean():.4f} std={pr_primary.std():.4f}')

    pr_by_residue = dict(zip(residue_ids, pr_primary))

    # -------- Load DMS --------
    print(f'[rbd] loading DMS from {dms_path}...', flush=True)
    dms = pd.read_csv(dms_path)
    print(f'  DMS columns: {list(dms.columns)}')
    print(f'  DMS shape: {dms.shape}')

    # Identify position and bind columns
    pos_col = None
    bind_col = None
    mut_col = None
    wt_col = None
    for c in dms.columns:
        lc = c.lower()
        # Prefer spike numbering (site_SARS2) since it matches PDB 6M0J numbering
        if pos_col is None and lc in ('site_sars2', 'site_sars-2'):
            pos_col = c
        if bind_col is None and 'bind' in lc and 'avg' in lc:
            bind_col = c
        if mut_col is None and lc in ('mutant', 'mut', 'mutation', 'mutant_aa', 'aa_mut'):
            mut_col = c
        if wt_col is None and lc in ('wildtype', 'wt', 'wildtype_aa', 'aa_wt'):
            wt_col = c
    # Fallback: site_RBD if site_SARS2 not found (different numbering)
    if pos_col is None:
        for c in dms.columns:
            lc = c.lower()
            if lc in ('site', 'position', 'pos', 'residue', 'site_rbd'):
                pos_col = c
                break
    if pos_col is None:
        for c in dms.columns:
            if dms[c].dtype in (int, np.int64, 'int64') and dms[c].min() > 100 and dms[c].max() < 600:
                pos_col = c; break
    if bind_col is None:
        for c in dms.columns:
            lc = c.lower()
            if 'bind' in lc or 'ace2' in lc:
                bind_col = c; break
    print(f'  detected: pos_col={pos_col}, bind_col={bind_col}, mut_col={mut_col}, wt_col={wt_col}')
    if pos_col is None or bind_col is None:
        raise RuntimeError('Cannot identify position/binding columns; manual inspection needed.')

    # Filter missense: remove stop codons
    if mut_col is not None:
        dms = dms[dms[mut_col] != '*']
        if wt_col is not None:
            dms = dms[dms[mut_col] != dms[wt_col]]
    dms = dms.dropna(subset=[pos_col, bind_col])

    # Per-residue outcome
    grouped = dms.groupby(pos_col)[bind_col].agg(
        lambda v: np.mean(np.abs(v))
    ).rename('mean_abs_bind')
    counts = dms.groupby(pos_col)[bind_col].count().rename('n_mutants')

    outcome = pd.concat([grouped, counts], axis=1).reset_index()
    outcome = outcome[outcome['n_mutants'] >= MIN_MUTANTS]
    print(f'  positions with >={MIN_MUTANTS} mutants: {len(outcome)}')

    # Merge with PR
    outcome['PR'] = outcome[pos_col].map(pr_by_residue)
    in_rbm = outcome[pos_col].between(RBM_RANGE[0], RBM_RANGE[1])
    outcome['in_RBM'] = in_rbm

    merged = outcome.dropna(subset=['PR']).copy()
    print(f'  positions with both PR and DMS: {len(merged)}')

    # -------- PRIMARY TEST --------
    r_primary = spearman_ci(merged['PR'].values, merged['mean_abs_bind'].values)
    print(f'\n=== PRIMARY TEST ===')
    print(f'  rho(PR, mean|delta_bind|) = {r_primary["rho"]:+.4f}')
    print(f'  CI [{r_primary["ci_lo"]:+.4f}, {r_primary["ci_hi"]:+.4f}]  p={r_primary["p"]:.2e}  n={r_primary["n"]}')

    # -------- METHODOLOGY NULL GATE --------
    print(f'\n=== METHODOLOGY NULL (n_perm={N_PERM}) ===')
    rng = np.random.default_rng(RNG_SEED + 1)
    x = merged['PR'].values
    y = merged['mean_abs_bind'].values
    nulls = np.empty(N_PERM)
    for i in range(N_PERM):
        nulls[i] = sp.spearmanr(x, rng.permutation(y)).statistic
    null_mean = float(nulls.mean())
    null_std = float(nulls.std())
    null_ci = (float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5)))
    null_span = max(abs(null_ci[0]), abs(null_ci[1]))
    print(f'  null mean={null_mean:+.4f}  std={null_std:.4f}  CI [{null_ci[0]:+.4f}, {null_ci[1]:+.4f}]')
    print(f'  span={null_span:.4f}')
    if null_span <= 0.10:
        gate = 'PASS'; th = {'strong': 0.30, 'partial': 0.20, 'weak': 0.10}
    elif null_span <= 0.15:
        gate = 'CAUTION'; th = {'strong': 0.35, 'partial': 0.25, 'weak': 0.15}
    else:
        gate = 'FAIL'; th = None
    print(f'  gate: {gate}')

    # -------- DECISION --------
    print(f'\n=== VERDICT ===')
    rho = r_primary['rho']; p = r_primary['p']; mag = abs(rho)
    if gate == 'FAIL':
        verdict = 'HOLD (methodology gate failed)'
    elif rho > 0 and mag >= 0.10:
        verdict = 'FAIL_WRONG_DIRECTION'
    elif mag >= th['strong'] and rho < 0 and p < 0.01:
        verdict = 'PASS'
    elif mag >= th['partial'] and rho < 0:
        verdict = 'PARTIAL'
    elif mag >= th['weak'] and rho < 0:
        verdict = 'WEAK'
    else:
        verdict = 'NULL'
    print(f'  >>> {verdict} <<<')

    # -------- SECONDARY: lambda sensitivity --------
    print(f'\n=== SECONDARY: lambda sensitivity (EXPLORATORY) ===')
    sensitivity = {}
    for lam in LAMBDA_SENSITIVITY:
        pr_lam = per_residue_PR(coords, lam)
        pr_lam_by_res = dict(zip(residue_ids, pr_lam))
        xlam = merged[[pos_col]].copy()
        xlam['PR_lam'] = xlam[pos_col].map(pr_lam_by_res)
        m = xlam.dropna().merge(merged[[pos_col, 'mean_abs_bind']], on=pos_col)
        rho_lam = sp.spearmanr(m['PR_lam'], m['mean_abs_bind']).statistic
        sensitivity[lam] = {'rho': float(rho_lam), 'n': int(len(m))}
        print(f'  lambda={lam}: rho={rho_lam:+.4f}  n={len(m)}')

    # -------- SECONDARY: RBM vs non-RBM --------
    print(f'\n=== SECONDARY: RBM vs non-RBM (EXPLORATORY) ===')
    stratified = {}
    for label, mask in [('RBM', merged['in_RBM']), ('non_RBM', ~merged['in_RBM'])]:
        sub = merged[mask]
        if len(sub) < 10:
            stratified[label] = {'n': int(len(sub)), 'rho': None}
            continue
        rho_s = sp.spearmanr(sub['PR'], sub['mean_abs_bind']).statistic
        p_s = sp.spearmanr(sub['PR'], sub['mean_abs_bind']).pvalue
        stratified[label] = {'n': int(len(sub)), 'rho': float(rho_s), 'p': float(p_s)}
        print(f'  {label}: n={len(sub)}  rho={rho_s:+.4f}  p={p_s:.3f}')

    # -------- Save --------
    merged_out = merged.rename(columns={pos_col: 'position'})
    out_csv = os.path.join(TABLES, 'biology_rbd_per_residue.csv')
    merged_out.to_csv(out_csv, index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(merged['PR'], merged['mean_abs_bind'], s=30,
               c=['#d62728' if b else '#1f77b4' for b in merged['in_RBM']],
               alpha=0.7, edgecolor='k', linewidth=0.4)
    ax.set_xlabel(f'contact-map PR (λ={LAMBDA_A} Å)')
    ax.set_ylabel('mean |Δbind_avg| across missense mutants')
    ax.set_title(f'Primary: ρ={rho:+.3f} CI [{r_primary["ci_lo"]:+.3f},{r_primary["ci_hi"]:+.3f}] n={r_primary["n"]}')
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=7, label='RBM'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=7, label='non-RBM')],
              ['RBM (437–507)', 'non-RBM'], fontsize=8)

    ax = axes[1]
    ax.hist(nulls, bins=40, color='#999999', alpha=0.7, label='methodology null')
    ax.axvline(rho, color='#d62728', linewidth=2, label=f'observed ρ={rho:+.3f}')
    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel('Spearman ρ'); ax.set_ylabel('count')
    ax.set_title(f'Methodology null (span={null_span:.3f}, gate={gate})')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'biology_rbd_scatter.pdf'))
    plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'pre_registration': 'invariants_preregistration_v4_rbd.md',
        'pre_registration_commit': commit,
        'substrate': 'SARS-CoV-2 RBD',
        'pdb': PDB_ID,
        'chain': PDB_CHAIN,
        'lambda_angstrom': LAMBDA_A,
        'min_mutants': MIN_MUTANTS,
        'primary': {'rho': r_primary, 'verdict': verdict},
        'methodology_null': {
            'mean': null_mean, 'std': null_std,
            'ci_95': null_ci, 'span': float(null_span),
            'gate': gate, 'adjusted_thresholds': th,
        },
        'secondary_lambda_sensitivity': sensitivity,
        'secondary_RBM_stratified': stratified,
        'n_residues_final': int(len(merged)),
    }
    out_json = os.path.join(HERE, 'biology_rbd_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out_csv}')
    print(f'wrote {out_json}')


if __name__ == '__main__':
    main()
