#!/usr/bin/env python3
"""
Validation script for run_overnight_analysis.py
Tests code structure, imports, and logic without requiring full database
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    results = {}

    # Core Python
    try:
        import sqlite3, time, gc
        from datetime import datetime
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        results['core_python'] = 'OK'
    except ImportError as e:
        results['core_python'] = f'FAIL: {e}'

    # Data science
    try:
        import pandas as pd
        import numpy as np
        results['pandas_numpy'] = 'OK'
    except ImportError as e:
        results['pandas_numpy'] = f'FAIL: {e}'

    # Visualization
    try:
        import matplotlib.pyplot as plt
        results['matplotlib'] = 'OK'
    except ImportError as e:
        results['matplotlib'] = f'FAIL: {e}'

    # Scipy
    try:
        from scipy import stats
        from scipy.ndimage import gaussian_filter1d
        results['scipy'] = 'OK'
    except ImportError as e:
        results['scipy'] = f'FAIL: {e}'

    # Sklearn
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import KMeans
        from sklearn.covariance import LedoitWolf
        from sklearn.utils import resample
        results['sklearn'] = 'OK'
    except ImportError as e:
        results['sklearn'] = f'FAIL: {e}'

    # HMM (optional)
    try:
        from hmmlearn.hmm import GaussianHMM
        results['hmmlearn'] = 'OK'
    except ImportError:
        results['hmmlearn'] = 'OPTIONAL - Not installed (HMM will be skipped)'

    for name, status in results.items():
        print(f"  {name:20}: {status}")

    return all('FAIL' not in v for v in results.values() if 'OPTIONAL' not in v)


def test_config():
    """Test configuration validity"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)

    CONFIG = {
        'years_back': 20,
        'window_sizes': [21, 42, 63, 126, 252],
        'step_size': 1,
        'n_bootstrap': 100,
        'bootstrap_confidence': 0.95,
        'n_monte_carlo': 500,
        'mc_block_size': 21,
        'n_regimes': 3,
        'consensus_thresholds': [0.3, 0.4, 0.5, 0.6, 0.7],
        'p_value_threshold': 0.05,
        'n_jobs': -1,
        'chunk_size': 100,
        'run_multiresolution': True,
        'run_bootstrap': True,
        'run_montecarlo': True,
        'run_hmm_analysis': True,
    }

    errors = []

    # Validate window sizes
    if not all(w > 0 for w in CONFIG['window_sizes']):
        errors.append("window_sizes must be positive")

    # Validate step size
    if CONFIG['step_size'] < 1:
        errors.append("step_size must be >= 1")

    # Validate bootstrap
    if CONFIG['n_bootstrap'] < 10:
        errors.append("n_bootstrap should be >= 10 for meaningful CIs")

    # Validate Monte Carlo
    if CONFIG['n_monte_carlo'] < 100:
        errors.append("n_monte_carlo should be >= 100 for good null distribution")

    # Validate thresholds
    if not all(0 < t < 1 for t in CONFIG['consensus_thresholds']):
        errors.append("consensus_thresholds must be between 0 and 1")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False

    print("  Configuration validation: PASSED")

    # Estimate runtime
    print("\n  Estimated runtime breakdown:")
    n_days = CONFIG['years_back'] * 252  # Trading days

    for ws in CONFIG['window_sizes']:
        n_windows = (n_days - ws) // CONFIG['step_size']
        print(f"    Window {ws:3d}d: ~{n_windows:,} lens computations")

    total_multireso = sum((n_days - ws) // CONFIG['step_size'] for ws in CONFIG['window_sizes'])
    total_bootstrap = (n_days - 63) // 5 * CONFIG['n_bootstrap']  # Step=5 for bootstrap
    total_montecarlo = CONFIG['n_monte_carlo'] * 50  # ~50 samples per sim

    print(f"\n  Total computations:")
    print(f"    Multi-resolution: ~{total_multireso:,} lens calcs")
    print(f"    Bootstrap:        ~{total_bootstrap:,} lens calcs")
    print(f"    Monte Carlo:      ~{total_montecarlo:,} lens calcs")
    print(f"    GRAND TOTAL:      ~{total_multireso + total_bootstrap + total_montecarlo:,}")

    return True


def test_lens_functions():
    """Test lens computation with synthetic data"""
    print("\n" + "=" * 60)
    print("TESTING LENS FUNCTIONS (Synthetic Data)")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("  SKIP: pandas/numpy not installed")
        return False

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    returns = pd.DataFrame(
        np.random.randn(n_samples, n_features) * 0.02,
        columns=[f'asset_{i}' for i in range(n_features)]
    )
    prices = (1 + returns).cumprod() * 100

    print(f"  Test data: {n_samples} samples x {n_features} features")

    # Test each lens individually
    lens_tests = {}

    # 1. PCA
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaled = StandardScaler().fit_transform(returns.values)
        pca = PCA(n_components=3)
        pca.fit(scaled)
        score = pca.explained_variance_ratio_[0]
        lens_tests['PCA'] = f'OK (score={score:.3f})'
    except Exception as e:
        lens_tests['PCA'] = f'FAIL: {e}'

    # 2. Correlation
    try:
        corr = np.corrcoef(returns.values.T)
        upper = corr[np.triu_indices(len(corr), k=1)]
        score = np.mean(np.abs(upper))
        lens_tests['Correlation'] = f'OK (score={score:.3f})'
    except Exception as e:
        lens_tests['Correlation'] = f'FAIL: {e}'

    # 3. Volatility
    try:
        score = np.std(returns.values, axis=0).mean()
        lens_tests['Volatility'] = f'OK (score={score:.4f})'
    except Exception as e:
        lens_tests['Volatility'] = f'FAIL: {e}'

    # 4. Anomaly (IsolationForest)
    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(StandardScaler().fit_transform(returns.values))
        score = (preds == -1).mean()
        lens_tests['Anomaly'] = f'OK (score={score:.3f})'
    except Exception as e:
        lens_tests['Anomaly'] = f'FAIL: {e}'

    # 5. GMM Regime
    try:
        from sklearn.mixture import GaussianMixture
        mean_ret = returns.values.mean(axis=1).reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(mean_ret)
        probs = gmm.predict_proba(mean_ret)
        score = probs[:, np.argmin(gmm.means_.flatten())].mean()
        lens_tests['Regime_GMM'] = f'OK (score={score:.3f})'
    except Exception as e:
        lens_tests['Regime_GMM'] = f'FAIL: {e}'

    # 6. Covariance Regime (Ledoit-Wolf)
    try:
        from sklearn.covariance import LedoitWolf
        mid = n_samples // 2
        lw1 = LedoitWolf().fit(returns.values[:mid])
        lw2 = LedoitWolf().fit(returns.values[mid:])
        diff = np.linalg.norm(lw1.covariance_ - lw2.covariance_, 'fro')
        avg = (np.linalg.norm(lw1.covariance_, 'fro') +
               np.linalg.norm(lw2.covariance_, 'fro')) / 2
        score = diff / avg
        lens_tests['Covariance'] = f'OK (score={score:.3f})'
    except Exception as e:
        lens_tests['Covariance'] = f'FAIL: {e}'

    # 7. HMM
    try:
        from hmmlearn.hmm import GaussianHMM
        mean_ret = returns.values.mean(axis=1).reshape(-1, 1)
        hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=42)
        hmm.fit(mean_ret)
        states = hmm.predict(mean_ret)
        transitions = np.sum(states[1:] != states[:-1])
        score = transitions / len(states)
        lens_tests['HMM'] = f'OK (score={score:.3f})'
    except ImportError:
        lens_tests['HMM'] = 'SKIP (hmmlearn not installed)'
    except Exception as e:
        lens_tests['HMM'] = f'FAIL: {e}'

    for name, status in lens_tests.items():
        print(f"  {name:15}: {status}")

    failures = sum(1 for v in lens_tests.values() if v.startswith('FAIL'))
    return failures == 0


def test_database():
    """Test database connectivity"""
    print("\n" + "=" * 60)
    print("TESTING DATABASE")
    print("=" * 60)

    import sqlite3
    from pathlib import Path

    db_path = Path.home() / "prism_data" / "prism.db"

    if not db_path.exists():
        print(f"  Database not found: {db_path}")
        print("  SKIP: Cannot test without database")
        return None

    try:
        conn = sqlite3.connect(db_path)

        # Check tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor]
        print(f"  Tables found: {tables}")

        # Check data counts
        for table in ['market_prices', 'econ_values']:
            if table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count:,} rows")

        conn.close()
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_output_directory():
    """Test output directory"""
    print("\n" + "=" * 60)
    print("TESTING OUTPUT DIRECTORY")
    print("=" * 60)

    from pathlib import Path

    output_dir = PROJECT_ROOT / "output" / "overnight_analysis"

    if output_dir.exists():
        files = list(output_dir.glob("*"))
        print(f"  Output directory exists: {output_dir}")
        print(f"  Files present: {len(files)}")
        for f in files[:5]:
            print(f"    - {f.name}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")
        return True
    else:
        print(f"  Output directory will be created: {output_dir}")
        # Test if we can create it
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print("  Directory creation: OK")
            return True
        except Exception as e:
            print(f"  ERROR: Cannot create directory: {e}")
            return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("PRISM OVERNIGHT ANALYSIS - VALIDATION SUITE")
    print("=" * 60)

    results = {}

    results['imports'] = test_imports()
    results['config'] = test_config()
    results['lenses'] = test_lens_functions()
    results['database'] = test_database()
    results['output'] = test_output_directory()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    can_run = True
    for name, status in results.items():
        if status is None:
            status_str = "SKIP"
        elif status:
            status_str = "PASS"
        else:
            status_str = "FAIL"
            if name in ['imports', 'database']:
                can_run = False
        print(f"  {name:15}: {status_str}")

    print("\n" + "=" * 60)
    if can_run and results['database'] is not None:
        print("READY TO RUN overnight analysis")
        print("Command: python start/run_overnight_analysis.py")
    elif results['database'] is None:
        print("MISSING: Database required at ~/prism_data/prism.db")
        print("Create database first, then run overnight analysis")
    else:
        print("FIX ERRORS before running overnight analysis")
        print("Install missing dependencies with:")
        print("  pip install pandas numpy scipy scikit-learn matplotlib hmmlearn")
    print("=" * 60)


if __name__ == "__main__":
    main()
