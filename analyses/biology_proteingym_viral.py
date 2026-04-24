"""Pre-registered per-protein viral test on ProteinGym (pre-reg v5).

Locks: invariants_preregistration_v5_proteingym_viral.md.
One-shot. No rescue if FAILS.

Primary: rho(contact_PR, mean|DMS_score|) across viral proteins, predicted NEGATIVE.
"""
import json, os, sys, subprocess, urllib.request
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

PRE_REG_FILE = os.path.join(HERE, 'invariants_preregistration_v5_proteingym_viral.md')

# LOCKED parameters
LAMBDA_A = 8.0
MIN_MUTANTS = 100
RNG_SEED = 20260425
N_BOOT = 10_000
N_PERM = 1_000
LAMBDA_SENSITIVITY = [6.0, 10.0, 12.0]

# Data URLs
PG_REFERENCE_URL = 'https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv'
PG_PARQUET_TEMPLATE = 'https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-0000{}-of-00005.parquet'
N_PARQUET_CHUNKS = 5
AF_URL = 'https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb'
PG_STRUCT_ZIP_URL = 'https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/ProteinGym_AF2_structures.zip'


def get_pre_reg_commit():
    try:
        out = subprocess.run(['git', 'log', '-n', '1', '--pretty=format:%H', '--', PRE_REG_FILE],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()[:10]
    except Exception:
        return 'unknown'


def download(url, dest, min_size=1000, max_size=None):
    if os.path.exists(dest):
        sz = os.path.getsize(dest)
        if sz >= min_size and (max_size is None or sz <= max_size):
            return True
        os.remove(dest)
    try:
        urllib.request.urlretrieve(url, dest)
        sz = os.path.getsize(dest)
        return sz >= min_size
    except Exception as e:
        print(f'  download failed {url}: {e}', flush=True)
        return False


def parse_pdb_ca(pdb_path, chain=None):
    ca = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'): continue
            if chain is not None and line[21:22] != chain: continue
            if line[12:16].strip() != 'CA': continue
            altloc = line[16:17].strip()
            if altloc and altloc != 'A': continue
            resnum = int(line[22:26].strip())
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            if resnum not in ca:
                ca[resnum] = np.array([x, y, z])
    return ca


def contact_PR_from_coords(coords, lam=LAMBDA_A):
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    C = np.exp(-d / lam)
    sigma = np.linalg.svd(C, compute_uv=False)
    s2 = sigma ** 2
    denom = (s2 ** 2).sum()
    if denom <= 0:
        return np.nan
    return float(s2.sum() ** 2 / denom)


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
    print(f'[pg] pre-reg commit: {commit}', flush=True)

    cache = os.path.join(ROOT, 'data', 'proteingym_cache')
    af_cache = os.path.join(cache, 'alphafold')
    os.makedirs(af_cache, exist_ok=True)

    # -------- Reference --------
    ref_path = os.path.join(cache, 'DMS_substitutions.csv')
    print(f'[pg] fetching reference...', flush=True)
    download(PG_REFERENCE_URL, ref_path)
    ref = pd.read_csv(ref_path)
    viral = ref[ref['taxon'] == 'Virus'].copy()
    print(f'  ref: {len(ref)} total assays, viral subset: {len(viral)}')

    # -------- DMS parquet --------
    print(f'[pg] fetching DMS parquet chunks...', flush=True)
    dfs = []
    for i in range(N_PARQUET_CHUNKS):
        path = os.path.join(cache, f'train-0000{i}-of-00005.parquet')
        download(PG_PARQUET_TEMPLATE.format(i), path)
        dfs.append(pd.read_parquet(path))
    dms = pd.concat(dfs, ignore_index=True)
    print(f'  total rows: {len(dms)}  unique assays: {dms.DMS_id.nunique()}')

    # -------- Per-assay viral filter --------
    viral_ids = set(viral['DMS_id'].values)
    dms_viral = dms[dms['DMS_id'].isin(viral_ids)].copy()
    print(f'  viral rows: {len(dms_viral)}  assays: {dms_viral.DMS_id.nunique()}')

    # -------- Per-protein DMS aggregate --------
    agg = dms_viral.groupby('DMS_id').agg(
        mean_abs_DMS=('DMS_score', lambda v: np.mean(np.abs(v))),
        frac_unfit=('DMS_score_bin', lambda v: (v == 0).mean()),
        n_mutants=('DMS_score', 'count'),
    ).reset_index()
    # Merge with reference for UniProt_ID, taxon
    agg = agg.merge(viral[['DMS_id', 'UniProt_ID', 'seq_len', 'coarse_selection_type']], on='DMS_id')
    # Coverage filter
    before = len(agg)
    agg = agg[agg['n_mutants'] >= MIN_MUTANTS].copy()
    print(f'  {before} -> {len(agg)} proteins after MIN_MUTANTS={MIN_MUTANTS}')

    # -------- ProteinGym-bundled AF2 structures (already sliced to DMS region) --------
    struct_zip = os.path.join(cache, 'ProteinGym_AF2_structures.zip')
    struct_dir = os.path.join(cache, 'af2_structures', 'ProteinGym_AF2_structures')
    if not os.path.exists(struct_dir) or len(os.listdir(struct_dir)) < 10:
        print(f'[pg] downloading ProteinGym AF2 structures ({PG_STRUCT_ZIP_URL})...', flush=True)
        download(PG_STRUCT_ZIP_URL, struct_zip, min_size=10_000_000)
        import zipfile
        with zipfile.ZipFile(struct_zip) as zf:
            zf.extractall(os.path.join(cache, 'af2_structures'))
    print(f'[pg] structures ready: {len(os.listdir(struct_dir))} PDB files', flush=True)

    print(f'[pg] computing contact_PR per viral protein...', flush=True)
    results = []
    for _, row in agg.iterrows():
        uid = row['UniProt_ID']
        pdb_path = os.path.join(struct_dir, f'{uid}.pdb')
        if not os.path.exists(pdb_path):
            results.append({**row, 'contact_PR': np.nan, 'n_residues': 0,
                             'status': 'pdb_missing'})
            continue
        ca = parse_pdb_ca(pdb_path)
        if len(ca) < 20:
            results.append({**row, 'contact_PR': np.nan, 'n_residues': len(ca),
                             'status': 'too_short'})
            continue
        coords = np.array([ca[k] for k in sorted(ca.keys())])
        pr = contact_PR_from_coords(coords, LAMBDA_A)
        results.append({**row, 'contact_PR': pr, 'n_residues': len(ca),
                         'status': 'ok'})
        print(f'  {row["DMS_id"][:35]:<35}  n_res={len(ca):>5}  PR={pr:>7.2f}  n_mut={int(row["n_mutants"]):>6}', flush=True)

    table = pd.DataFrame(results)
    usable = table[table['status'] == 'ok'].copy()
    print(f'\nUsable proteins: {len(usable)} / {len(agg)}')
    if len(usable) < 8:
        print('WARNING: n too small for meaningful correlation. Reporting as-is.')

    # -------- PRIMARY --------
    print(f'\n=== PRIMARY: rho(contact_PR, mean|DMS_score|) ===')
    r_primary = spearman_ci(usable['contact_PR'].values, usable['mean_abs_DMS'].values)
    if r_primary is None:
        print('  NO_DATA')
        verdict = 'NO_DATA'; gate = 'NO_DATA'; th = None
        null_mean = null_std = null_span = None; null_ci = (None, None)
        null_rhos = np.array([])
    else:
        print(f'  rho={r_primary["rho"]:+.4f}  CI [{r_primary["ci_lo"]:+.4f}, {r_primary["ci_hi"]:+.4f}]  '
              f'p={r_primary["p"]:.3e}  n={r_primary["n"]}')

        # -------- METHODOLOGY NULL --------
        print(f'\n=== METHODOLOGY NULL ({N_PERM} perms) ===')
        rng = np.random.default_rng(RNG_SEED + 1)
        x = usable['contact_PR'].values; y = usable['mean_abs_DMS'].values
        null_rhos = np.empty(N_PERM)
        for i in range(N_PERM):
            null_rhos[i] = sp.spearmanr(x, rng.permutation(y)).statistic
        null_rhos = null_rhos[np.isfinite(null_rhos)]
        null_mean = float(null_rhos.mean()); null_std = float(null_rhos.std())
        null_ci = (float(np.percentile(null_rhos, 2.5)), float(np.percentile(null_rhos, 97.5)))
        null_span = max(abs(null_ci[0]), abs(null_ci[1]))
        print(f'  null mean={null_mean:+.4f}  std={null_std:.4f}  CI [{null_ci[0]:+.4f}, {null_ci[1]:+.4f}]  span={null_span:.4f}')
        if null_span <= 0.10:
            gate = 'PASS'; th = {'strong': 0.30, 'partial': 0.20, 'weak': 0.10}
        elif null_span <= 0.15:
            gate = 'CAUTION'; th = {'strong': 0.35, 'partial': 0.25, 'weak': 0.15}
        else:
            gate = 'FAIL'; th = None
        print(f'  gate: {gate}')

        # -------- VERDICT --------
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

    print(f'\n=== VERDICT: {verdict} ===')

    # -------- SECONDARY --------
    sensitivity = {}
    if len(usable) >= 8:
        print(f'\n=== SECONDARY: lambda sensitivity (exploratory) ===')
        for lam in LAMBDA_SENSITIVITY:
            pr_lam = []
            for _, row in usable.iterrows():
                uid = row['UniProt_ID']
                pdb_path = os.path.join(struct_dir, f'{uid}.pdb')
                ca = parse_pdb_ca(pdb_path)
                coords = np.array([ca[k] for k in sorted(ca.keys())])
                pr_lam.append(contact_PR_from_coords(coords, lam))
            pr_lam = np.array(pr_lam)
            rho_lam = sp.spearmanr(pr_lam, usable['mean_abs_DMS']).statistic
            sensitivity[lam] = float(rho_lam)
            print(f'  lambda={lam}: rho={rho_lam:+.4f}')

    # Binary outcome secondary
    binary = None
    if len(usable) >= 8:
        print(f'\n=== SECONDARY: binary outcome (exploratory) ===')
        r_bin = spearman_ci(usable['contact_PR'].values, usable['frac_unfit'].values, n_boot=2000)
        print(f'  rho(contact_PR, frac_unfit) = {r_bin["rho"]:+.4f}  p={r_bin["p"]:.3f}')
        binary = r_bin

    # Split by selection type
    by_selection = {}
    if len(usable) >= 8:
        print(f'\n=== SECONDARY: by coarse_selection_type ===')
        for sel, sub in usable.groupby('coarse_selection_type'):
            if len(sub) < 5:
                by_selection[str(sel)] = {'n': len(sub), 'rho': None}
                continue
            rho_s = sp.spearmanr(sub['contact_PR'], sub['mean_abs_DMS']).statistic
            by_selection[str(sel)] = {'n': int(len(sub)), 'rho': float(rho_s)}
            print(f'  {sel}: n={len(sub)}  rho={rho_s:+.4f}')

    # -------- Save --------
    out_csv = os.path.join(TABLES, 'biology_proteingym_viral.csv')
    table.to_csv(out_csv, index=False)

    # Plot
    if len(usable) >= 8:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax = axes[0]
        colors_map = {'Activity': '#1f77b4', 'OrganismalFitness': '#d62728',
                      'Binding': '#2ca02c', 'Expression': '#ff7f0e', 'Stability': '#9467bd'}
        for sel, sub in usable.groupby('coarse_selection_type'):
            ax.scatter(sub['contact_PR'], sub['mean_abs_DMS'],
                       s=70, alpha=0.8, color=colors_map.get(sel, '#999999'),
                       label=f'{sel} n={len(sub)}', edgecolor='k', linewidth=0.5)
        for _, row in usable.iterrows():
            lbl = row['DMS_id'].split('_')[0]
            ax.annotate(lbl, (row['contact_PR'], row['mean_abs_DMS']),
                         fontsize=6, xytext=(2, 2), textcoords='offset points', alpha=0.7)
        ax.set_xlabel(f'contact-PR (λ={LAMBDA_A} Å)')
        ax.set_ylabel('mean |DMS_score|')
        rho_t = r_primary['rho'] if r_primary else np.nan
        ax.set_title(f'Primary: ρ={rho_t:+.3f}  n={len(usable)}  verdict={verdict}')
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

        ax = axes[1]
        if len(null_rhos) > 0:
            ax.hist(null_rhos, bins=40, alpha=0.7, color='#999999', label='null')
            ax.axvline(rho_t, color='#d62728', linewidth=2, label=f'obs={rho_t:+.3f}')
        ax.axvline(0, color='k', linestyle='--', alpha=0.4)
        ax.set_xlabel('ρ'); ax.set_ylabel('count')
        ax.set_title(f'Methodology null gate: {gate}')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, 'biology_proteingym_viral.pdf'))
        plt.close(fig)

    summary = {
        'generated': pd.Timestamp.utcnow().isoformat(),
        'pre_registration': 'invariants_preregistration_v5_proteingym_viral.md',
        'pre_registration_commit': commit,
        'substrate': 'ProteinGym v1 viral DMS subset',
        'lambda_angstrom': LAMBDA_A,
        'min_mutants': MIN_MUTANTS,
        'n_proteins_ref': int(len(viral)),
        'n_proteins_usable': int(len(usable)),
        'primary': {'rho': r_primary, 'verdict': verdict},
        'methodology_null': {
            'mean': null_mean, 'std': null_std,
            'ci_95': null_ci, 'span': float(null_span) if null_span is not None else None,
            'gate': gate, 'adjusted_thresholds': th,
        },
        'secondary_lambda_sensitivity': sensitivity,
        'secondary_binary_outcome': binary,
        'secondary_by_selection': by_selection,
    }
    out_json = os.path.join(HERE, 'biology_proteingym_viral_summary.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nwrote {out_csv}')
    print(f'wrote {out_json}')


if __name__ == '__main__':
    main()
