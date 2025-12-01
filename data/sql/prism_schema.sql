-- ============================================================================
-- PRISM Engine SQL Schema
-- ============================================================================
-- This schema handles BOTH input data AND engine outputs.
-- Location: /MyDrive/prismsql/prism.db
-- ============================================================================

-- ============================================================================
-- SECTION 1: INPUT DATA (Raw indicator time series)
-- ============================================================================

-- Master list of all indicators
-- Each indicator has a panel type, frequency, and source
CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,           -- e.g., 'SPY', 'DXY', 'WALCL'
    panel TEXT NOT NULL,                  -- e.g., 'equity', 'rates', 'liquidity', 'currency'
    frequency TEXT NOT NULL DEFAULT 'daily',  -- 'daily', 'weekly', 'monthly', 'quarterly'
    units TEXT,                           -- e.g., 'USD', 'percent', 'index', 'billions'
    source TEXT,                          -- e.g., 'FRED', 'Yahoo', 'manual'
    description TEXT,                     -- Human-readable description
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Time series values for each indicator
CREATE TABLE IF NOT EXISTS indicator_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,
    date TEXT NOT NULL,                   -- ISO format: 'YYYY-MM-DD'
    value REAL,                           -- The actual data point
    value_2 REAL,                         -- Optional second value (e.g., volume, spread)
    adjusted_value REAL,                  -- For adjusted prices, inflation-adjusted, etc.
    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
    UNIQUE(indicator_id, date)            -- Prevent duplicate entries
);

-- Index for fast date range queries (CRITICAL for performance)
CREATE INDEX IF NOT EXISTS idx_values_date ON indicator_values(date);
CREATE INDEX IF NOT EXISTS idx_values_indicator_date ON indicator_values(indicator_id, date);


-- ============================================================================
-- SECTION 2: ENGINE CONFIGURATION
-- ============================================================================

-- List of analytical lenses available in PRISM
CREATE TABLE IF NOT EXISTS lenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,            -- e.g., 'pca', 'granger', 'dmd', 'network'
    description TEXT,
    category TEXT                         -- 'correlation', 'causality', 'spectral', 'network'
);

-- Pre-populate the 14 lenses (run once)
INSERT OR IGNORE INTO lenses (name, description, category) VALUES
    ('pca', 'Principal Component Analysis - variance explained', 'dimensionality'),
    ('granger', 'Granger Causality - predictive relationships', 'causality'),
    ('transfer_entropy', 'Transfer Entropy - information flow', 'causality'),
    ('dmd', 'Dynamic Mode Decomposition - spectral analysis', 'spectral'),
    ('wavelet', 'Wavelet Coherence - multi-scale relationships', 'spectral'),
    ('correlation', 'Pearson Correlation - linear relationships', 'correlation'),
    ('mutual_info', 'Mutual Information - nonlinear dependence', 'correlation'),
    ('network', 'Network Centrality - graph importance', 'network'),
    ('cointegration', 'Cointegration - long-run equilibrium', 'equilibrium'),
    ('var_decomp', 'VAR Forecast Error Decomposition', 'causality'),
    ('rolling_beta', 'Rolling Beta - time-varying sensitivity', 'sensitivity'),
    ('partial_corr', 'Partial Correlation - conditional relationships', 'correlation'),
    ('cross_correlation', 'Cross-Correlation - lagged relationships', 'correlation'),
    ('copula', 'Copula Dependence - tail relationships', 'dependence');


-- ============================================================================
-- SECTION 3: TEMPORAL ANALYSIS WINDOWS
-- ============================================================================

-- Define analysis windows (e.g., 5-year rolling periods)
CREATE TABLE IF NOT EXISTS windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_date TEXT NOT NULL,             -- 'YYYY-MM-DD'
    end_date TEXT NOT NULL,               -- 'YYYY-MM-DD'
    start_year INTEGER,                   -- For quick filtering
    end_year INTEGER,
    increment INTEGER,                    -- How many years this window shifted
    n_observations INTEGER,               -- Actual data points in window
    label TEXT,                           -- e.g., '1970-1975', 'post_covid'
    UNIQUE(start_date, end_date)
);


-- ============================================================================
-- SECTION 4: ENGINE OUTPUTS
-- ============================================================================

-- Per-lens results for each indicator in each window
CREATE TABLE IF NOT EXISTS lens_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    indicator_id INTEGER NOT NULL,
    lens_id INTEGER NOT NULL,
    rank REAL,                            -- Rank within this window/lens
    raw_score REAL,                       -- The actual metric value
    normalized_score REAL,                -- 0-1 normalized score
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (window_id) REFERENCES windows(id),
    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
    FOREIGN KEY (lens_id) REFERENCES lenses(id),
    UNIQUE(window_id, indicator_id, lens_id)
);

-- Consensus rankings (aggregated across lenses)
CREATE TABLE IF NOT EXISTS consensus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id INTEGER NOT NULL,
    indicator_id INTEGER NOT NULL,
    avg_rank REAL,                        -- Mean rank across all lenses
    median_rank REAL,                     -- Median rank (more robust)
    std_rank REAL,                        -- Standard deviation (agreement measure)
    min_rank REAL,                        -- Best rank achieved
    max_rank REAL,                        -- Worst rank achieved
    agreement_score REAL,                 -- Custom coherence metric
    computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (window_id) REFERENCES windows(id),
    FOREIGN KEY (indicator_id) REFERENCES indicators(id),
    UNIQUE(window_id, indicator_id)
);

-- Indexes for fast output queries
CREATE INDEX IF NOT EXISTS idx_consensus_window ON consensus(window_id);
CREATE INDEX IF NOT EXISTS idx_consensus_rank ON consensus(avg_rank);
CREATE INDEX IF NOT EXISTS idx_lens_results_window ON lens_results(window_id);


-- ============================================================================
-- SECTION 5: REGIME ANALYSIS
-- ============================================================================

-- Track stability between consecutive windows
CREATE TABLE IF NOT EXISTS regime_stability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_before_id INTEGER NOT NULL,
    window_after_id INTEGER NOT NULL,
    transition_year INTEGER,              -- Year of the transition
    spearman_corr REAL,                   -- Rank correlation between periods
    kendall_tau REAL,                     -- Alternative correlation measure
    top_10_overlap INTEGER,               -- How many top-10 indicators stayed
    regime_break_flag INTEGER DEFAULT 0,  -- 1 if this is a significant break
    notes TEXT,                           -- e.g., 'COVID shock', 'Volcker era'
    FOREIGN KEY (window_before_id) REFERENCES windows(id),
    FOREIGN KEY (window_after_id) REFERENCES windows(id),
    UNIQUE(window_before_id, window_after_id)
);

-- Coherence snapshots (when lenses agree)
CREATE TABLE IF NOT EXISTS coherence_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    window_id INTEGER,
    coherence_score REAL,                 -- How much do lenses agree?
    n_lenses_agreeing INTEGER,            -- Count of aligned lenses
    top_indicator_id INTEGER,             -- Which indicator dominated
    event_type TEXT,                      -- 'convergence', 'divergence', 'regime_shift'
    description TEXT,
    FOREIGN KEY (window_id) REFERENCES windows(id),
    FOREIGN KEY (top_indicator_id) REFERENCES indicators(id)
);


-- ============================================================================
-- SECTION 6: RUN METADATA
-- ============================================================================

-- Track each engine run for reproducibility
CREATE TABLE IF NOT EXISTS engine_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    engine_version TEXT,                  -- e.g., 'PRISM 1.0', 'Phase III'
    config_json TEXT,                     -- Full configuration as JSON
    input_indicators TEXT,                -- Comma-separated list
    window_config TEXT,                   -- e.g., '5-year rolling, 1-year increment'
    lenses_used TEXT,                     -- Which lenses were run
    status TEXT DEFAULT 'running',        -- 'running', 'completed', 'failed'
    notes TEXT
);


-- ============================================================================
-- USEFUL VIEWS (pre-built queries you can use like tables)
-- ============================================================================

-- View: Current top indicators by consensus rank
CREATE VIEW IF NOT EXISTS v_top_indicators AS
SELECT 
    i.name AS indicator,
    i.panel,
    c.avg_rank,
    c.agreement_score,
    w.label AS window
FROM consensus c
JOIN indicators i ON c.indicator_id = i.id
JOIN windows w ON c.window_id = w.id
ORDER BY w.end_date DESC, c.avg_rank ASC;

-- View: Regime transitions summary
CREATE VIEW IF NOT EXISTS v_regime_transitions AS
SELECT 
    rs.transition_year,
    rs.spearman_corr,
    rs.top_10_overlap,
    CASE WHEN rs.regime_break_flag = 1 THEN 'BREAK' ELSE 'stable' END AS status,
    rs.notes
FROM regime_stability rs
ORDER BY rs.transition_year;

-- View: Indicator history across all windows
CREATE VIEW IF NOT EXISTS v_indicator_history AS
SELECT 
    i.name AS indicator,
    w.start_year,
    w.end_year,
    c.avg_rank,
    c.std_rank
FROM consensus c
JOIN indicators i ON c.indicator_id = i.id
JOIN windows w ON c.window_id = w.id
ORDER BY i.name, w.start_year;
