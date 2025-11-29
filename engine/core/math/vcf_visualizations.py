# VCF Research: Complete Visualization Suite
# Advanced plotting for vector variance, harmonic coherence, and regime analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, spectrogram
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: VECTOR VARIANCE VISUALIZATIONS
# ============================================================================

class VectorVisualizations:
    """
    Visualize vector dynamics in macro-financial space
    """
    
    @staticmethod
    def plot_vector_trajectory_3d(panel_df: pd.DataFrame, cols: list = None):
        """
        3D trajectory of the system in phase space
        Shows how the macro state evolves over time
        """
        if cols is None:
            cols = panel_df.columns[:3] if len(panel_df.columns) >= 3 else panel_df.columns
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract 3D coordinates
        x = panel_df[cols[0]].values
        y = panel_df[cols[1]].values
        z = panel_df[cols[2]].values if len(cols) >= 3 else np.zeros_like(x)
        
        # Color by time
        colors = np.arange(len(x))
        
        # Plot trajectory
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=20, alpha=0.6)
        ax.plot(x, y, z, 'gray', alpha=0.3, linewidth=0.5)
        
        # Mark start and end
        ax.scatter([x[0]], [y[0]], [z[0]], c='green', s=200, marker='o', 
                  edgecolors='black', linewidths=2, label='Start')
        ax.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=200, marker='X', 
                  edgecolors='black', linewidths=2, label='End')
        
        ax.set_xlabel(cols[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(cols[1], fontsize=12, fontweight='bold')
        ax.set_zlabel(cols[2] if len(cols) >= 3 else 'Z', fontsize=12, fontweight='bold')
        ax.set_title('Vector Trajectory in Phase Space', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Time', fontsize=11)
        
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_magnitude_evolution(vector_variance_results: dict, panel_df: pd.DataFrame):
        """
        Show how vector magnitude changes over time
        """
        # Compute magnitudes
        vectors = panel_df.values
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Time series
        ax1.plot(panel_df.index, magnitudes, linewidth=2, color='steelblue')
        mean_mag = vector_variance_results['magnitude_variance']['mean_magnitude']
        std_mag = vector_variance_results['magnitude_variance']['std_magnitude']
        
        ax1.axhline(mean_mag, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_mag:.2f}')
        ax1.fill_between(panel_df.index, mean_mag - std_mag, mean_mag + std_mag, 
                        alpha=0.2, color='red', label=f'±1 STD: {std_mag:.2f}')
        
        ax1.set_ylabel('Vector Magnitude', fontsize=12, fontweight='bold')
        ax1.set_title('System Magnitude Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(magnitudes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(mean_mag, color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(mean_mag - std_mag, color='orange', linestyle=':', linewidth=2, label='-1 STD')
        ax2.axvline(mean_mag + std_mag, color='orange', linestyle=':', linewidth=2, label='+1 STD')
        
        ax2.set_xlabel('Magnitude', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of System Magnitude', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_angular_dynamics(vector_variance_results: dict, panel_df: pd.DataFrame):
        """
        Visualize rotational dynamics
        """
        vectors = panel_df.values
        
        # Compute angles between consecutive vectors
        angles = []
        for i in range(len(vectors) - 1):
            v1 = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-10)
            v2 = vectors[i+1] / (np.linalg.norm(vectors[i+1]) + 1e-10)
            cos_angle = np.clip(np.dot(v1, v2), -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(np.degrees(angle))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Angular change over time
        ax1.plot(panel_df.index[1:], angles, linewidth=1.5, color='darkgreen', alpha=0.7)
        mean_angle = np.mean(angles)
        ax1.axhline(mean_angle, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_angle:.2f}°')
        
        ax1.set_ylabel('Angular Change (degrees)', fontsize=12, fontweight='bold')
        ax1.set_title('Rotational Velocity', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rotation
        cumulative_rotation = np.cumsum(angles)
        ax2.plot(panel_df.index[1:], cumulative_rotation, linewidth=2, color='purple')
        
        total_rot = vector_variance_results['directional_variance']['total_rotation']
        ax2.set_ylabel('Cumulative Rotation (degrees)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Total System Rotation: {np.degrees(total_rot):.1f}°', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_divergence_curl(vector_variance_results: dict):
        """
        Plot vector field divergence and curl
        """
        divergence = vector_variance_results['divergence']
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Divergence
        ax.plot(divergence.index, divergence.values, linewidth=2, color='crimson', label='Divergence')
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.fill_between(divergence.index, 0, divergence.values, 
                        where=(divergence.values > 0), alpha=0.3, color='red', label='Expansion')
        ax.fill_between(divergence.index, 0, divergence.values, 
                        where=(divergence.values < 0), alpha=0.3, color='blue', label='Contraction')
        
        ax.set_ylabel('Divergence', fontsize=12, fontweight='bold')
        ax.set_title('Vector Field Divergence (Expansion/Contraction)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 2: HARMONIC COHERENCE VISUALIZATIONS
# ============================================================================

class HarmonicVisualizations:
    """
    Visualize frequency-domain relationships
    """
    
    @staticmethod
    def plot_phase_space(panel_df: pd.DataFrame, col_a: str, col_b: str):
        """
        2D phase space with phase angles
        """
        # Extract phases
        signal_a = panel_df[col_a].values
        signal_b = panel_df[col_b].values
        
        phase_a = np.angle(hilbert(signal_a))
        phase_b = np.angle(hilbert(signal_b))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Phase space (signal values)
        colors = np.arange(len(signal_a))
        scatter = ax1.scatter(signal_a, signal_b, c=colors, cmap='plasma', s=30, alpha=0.6)
        ax1.plot(signal_a, signal_b, 'gray', alpha=0.2, linewidth=0.5)
        
        ax1.set_xlabel(col_a, fontsize=12, fontweight='bold')
        ax1.set_ylabel(col_b, fontsize=12, fontweight='bold')
        ax1.set_title('Signal Phase Space', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Time', fontsize=10)
        
        # Phase angle space
        scatter2 = ax2.scatter(phase_a, phase_b, c=colors, cmap='plasma', s=30, alpha=0.6)
        ax2.plot(phase_a, phase_b, 'gray', alpha=0.2, linewidth=0.5)
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta)*np.pi, np.sin(theta)*np.pi, 'k--', alpha=0.3, linewidth=1)
        
        ax2.set_xlabel(f'{col_a} Phase (rad)', fontsize=12, fontweight='bold')
        ax2.set_ylabel(f'{col_b} Phase (rad)', fontsize=12, fontweight='bold')
        ax2.set_title('Phase Angle Space', fontsize=14, fontweight='bold')
        ax2.set_xlim([-np.pi, np.pi])
        ax2.set_ylim([-np.pi, np.pi])
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Time', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_spectrograms(panel_df: pd.DataFrame, col_a: str, col_b: str, fs: float = 12):
        """
        Time-frequency analysis via spectrograms
        """
        signal_a = panel_df[col_a].values
        signal_b = panel_df[col_b].values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Spectrogram A
        f_a, t_a, Sxx_a = spectrogram(signal_a, fs=fs, nperseg=min(64, len(signal_a)//4))
        im1 = ax1.pcolormesh(t_a, f_a, 10 * np.log10(Sxx_a + 1e-10), shading='gouraud', cmap='hot')
        ax1.set_ylabel('Frequency (cycles/year)', fontsize=11, fontweight='bold')
        ax1.set_title(f'{col_a} - Time-Frequency Spectrogram', fontsize=13, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # Spectrogram B
        f_b, t_b, Sxx_b = spectrogram(signal_b, fs=fs, nperseg=min(64, len(signal_b)//4))
        im2 = ax2.pcolormesh(t_b, f_b, 10 * np.log10(Sxx_b + 1e-10), shading='gouraud', cmap='hot')
        ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency (cycles/year)', fontsize=11, fontweight='bold')
        ax2.set_title(f'{col_b} - Time-Frequency Spectrogram', fontsize=13, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_coherence_matrix(panel_df: pd.DataFrame, title: str = "Pairwise Coherence"):
        """
        Heatmap of all-pairs coherence
        """
        from scipy.signal import coherence
        
        cols = panel_df.columns
        n = len(cols)
        coh_matrix = np.zeros((n, n))
        
        # Compute coherence
        for i in range(n):
            for j in range(n):
                if i == j:
                    coh_matrix[i, j] = 1.0
                else:
                    f, Cxy = coherence(panel_df.iloc[:, i].values, 
                                      panel_df.iloc[:, j].values, 
                                      fs=1.0)
                    coh_matrix[i, j] = np.mean(Cxy)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(coh_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(cols, fontsize=10)
        
        # Add values
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{coh_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coherence', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_frequency_band_coherence(harmonic_results: dict):
        """
        Bar chart of coherence by frequency band
        """
        freq_bands = harmonic_results['frequency_bands']
        
        bands = list(freq_bands.keys())
        coherence = [freq_bands[b]['mean_coherence'] for b in bands]
        errors = [freq_bands[b]['coherence_std'] for b in bands]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(bands, coherence, yerr=errors, capsize=10, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel('Mean Coherence', fontsize=12, fontweight='bold')
        ax.set_title('Coherence by Frequency Band', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Add frequency ranges as text
        for i, (band, bar) in enumerate(zip(bands, bars)):
            freq_range = freq_bands[band]['freq_range']
            ax.text(i, bar.get_height() + 0.05, 
                   f'{freq_range[0]:.2f}-{freq_range[1]:.2f}',
                   ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 3: DMD VISUALIZATIONS
# ============================================================================

class DMDVisualizations:
    """
    Dynamic Mode Decomposition visualizations
    """
    
    @staticmethod
    def plot_eigenvalue_spectrum(dmd_results: dict):
        """
        Plot DMD eigenvalues on complex plane
        """
        eigenvalues = dmd_results['dmd_results']['eigenvalues']
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, alpha=0.5, label='Unit Circle')
        
        # Eigenvalues
        real = eigenvalues.real
        imag = eigenvalues.imag
        
        scatter = ax.scatter(real, imag, s=100, c=np.abs(eigenvalues), 
                           cmap='viridis', edgecolors='black', linewidths=2, alpha=0.8)
        
        # Annotate modes
        for i, (r, im) in enumerate(zip(real, imag)):
            ax.annotate(f'Mode {i}', (r, im), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Real Part', fontsize=12, fontweight='bold')
        ax.set_ylabel('Imaginary Part', fontsize=12, fontweight='bold')
        ax.set_title('DMD Eigenvalue Spectrum', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|λ| (Magnitude)', fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_dominant_modes(dmd_results: dict):
        """
        Visualize properties of dominant modes
        """
        dominant = dmd_results['dominant_modes']
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Amplitude
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['green' if s else 'red' for s in dominant['stable']]
        ax1.bar(range(len(dominant)), dominant['amplitude'], color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Mode Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
        ax1.set_title('Mode Amplitudes (Green=Stable, Red=Unstable)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Frequency
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(len(dominant)), dominant['frequency'], color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Mode Index', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency (cycles/period)', fontsize=11, fontweight='bold')
        ax2.set_title('Mode Frequencies', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Period
        ax3 = fig.add_subplot(gs[1, 0])
        periods = dominant['period'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(periods) > 0:
            ax3.bar(range(len(periods)), periods, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Mode Index', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Period (time units)', fontsize=11, fontweight='bold')
        ax3.set_title('Oscillation Periods', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Growth rate
        ax4 = fig.add_subplot(gs[1, 1])
        growth = dominant['growth_rate']
        colors4 = ['green' if g < 0 else 'red' for g in growth]
        ax4.bar(range(len(dominant)), growth, color=colors4, alpha=0.7, edgecolor='black')
        ax4.axhline(0, color='black', linewidth=2)
        ax4.set_xlabel('Mode Index', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Growth Rate', fontsize=11, fontweight='bold')
        ax4.set_title('Mode Growth Rates (Negative=Decay)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_reconstruction(panel_df: pd.DataFrame, dmd_results: dict, col: str = None):
        """
        Compare original vs DMD reconstruction
        """
        if col is None:
            col = panel_df.columns[0]
        
        reconstruction = dmd_results['reconstruction']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Original vs reconstruction
        ax1.plot(panel_df.index, panel_df[col], linewidth=2, label='Original', alpha=0.7)
        ax1.plot(reconstruction.index, reconstruction[col], linewidth=2, 
                label='DMD Reconstruction', linestyle='--', alpha=0.7)
        ax1.set_ylabel(col, fontsize=11, fontweight='bold')
        ax1.set_title(f'{col}: Original vs DMD Reconstruction', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Residual
        residual = panel_df[col].values - reconstruction[col].values
        ax2.plot(panel_df.index, residual, linewidth=1.5, color='red', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(panel_df.index, 0, residual, alpha=0.3, color='red')
        ax2.set_ylabel('Residual', fontsize=11, fontweight='bold')
        ax2.set_title('Reconstruction Error', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 4: REGIME DETECTION VISUALIZATIONS
# ============================================================================

class RegimeVisualizations:
    """
    Visualize regime changes
    """
    
    @staticmethod
    def plot_regime_timeline(regimes: pd.DataFrame, panel_df: pd.DataFrame):
        """
        Show regime changes over time with coherence
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Coherence over time
        ax1.plot(regimes.index, regimes['coherence'], linewidth=2, color='steelblue')
        ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
        ax1.fill_between(regimes.index, 0, regimes['coherence'], alpha=0.3, color='steelblue')
        
        # Mark regime changes
        change_dates = regimes[regimes['regime_change']].index
        for date in change_dates:
            ax1.axvline(date, color='red', linestyle='-', linewidth=2, alpha=0.7)
        
        ax1.set_ylabel('Mean Coherence', fontsize=11, fontweight='bold')
        ax1.set_title(f'Regime Detection Timeline ({len(change_dates)} changes detected)', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Show one series with regime backgrounds
        col = panel_df.columns[0]
        ax2.plot(panel_df.index, panel_df[col], linewidth=2, color='black')
        
        # Shade regime periods
        regime_id = 0
        colors_cycle = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        change_dates_list = [regimes.index[0]] + list(change_dates) + [regimes.index[-1]]
        for i in range(len(change_dates_list) - 1):
            ax2.axvspan(change_dates_list[i], change_dates_list[i+1], 
                       alpha=0.3, color=colors_cycle[regime_id % len(colors_cycle)])
            regime_id += 1
        
        ax2.set_ylabel(col, fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax2.set_title(f'{col} with Regime Backgrounds', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# PART 5: MASTER VISUALIZATION FUNCTION
# ============================================================================

def create_full_prism_report(panel_df: pd.DataFrame, 
                          vector_results: dict,
                          harmonic_results: dict,
                          dmd_results: dict,
                          regimes: pd.DataFrame,
                          col_a: str = None,
                          col_b: str = None):
    """
    Generate complete VCF visualization report
    """
    if col_a is None:
        col_a = panel_df.columns[0]
    if col_b is None:
        col_b = panel_df.columns[1] if len(panel_df.columns) > 1 else panel_df.columns[0]
    
    print("=" * 70)
    print("VCF RESEARCH - COMPLETE VISUALIZATION REPORT")
    print("=" * 70)
    
    print("\n1. Vector Trajectory (3D)")
    VectorVisualizations.plot_vector_trajectory_3d(panel_df)
    
    print("\n2. Magnitude Evolution")
    VectorVisualizations.plot_magnitude_evolution(vector_results, panel_df)
    
    print("\n3. Angular Dynamics")
    VectorVisualizations.plot_angular_dynamics(vector_results, panel_df)
    
    print("\n4. Divergence Analysis")
    VectorVisualizations.plot_divergence_curl(vector_results)
    
    print(f"\n5. Phase Space: {col_a} vs {col_b}")
    HarmonicVisualizations.plot_phase_space(panel_df, col_a, col_b)
    
    print(f"\n6. Spectrograms: {col_a} & {col_b}")
    HarmonicVisualizations.plot_spectrograms(panel_df, col_a, col_b)
    
    print("\n7. Coherence Matrix")
    HarmonicVisualizations.plot_coherence_matrix(panel_df)
    
    print("\n8. Frequency Band Coherence")
    HarmonicVisualizations.plot_frequency_band_coherence(harmonic_results)
    
    print("\n9. DMD Eigenvalue Spectrum")
    DMDVisualizations.plot_eigenvalue_spectrum(dmd_results)
    
    print("\n10. DMD Dominant Modes")
    DMDVisualizations.plot_dominant_modes(dmd_results)
    
    print(f"\n11. DMD Reconstruction: {col_a}")
    DMDVisualizations.plot_reconstruction(panel_df, dmd_results, col_a)
    
    print("\n12. Regime Detection Timeline")
    RegimeVisualizations.plot_regime_timeline(regimes, panel_df)
    
    print("\n" + "=" * 70)
    print("REPORT COMPLETE")
    print("=" * 70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Generate full report
    Assumes you already have results from VCFMathEngine
    """
    
    print("VCF Visualization Suite Loaded!")
    print("\nTo generate visualizations, run:")
    print("=" * 70)
    print("""
# Step 1: Run VCF analysis (from previous artifact)
from prism_math_models import VCFMathEngine

engine = VCFMathEngine(panel)
vector_results = engine.full_vector_analysis()
harmonic_results = engine.full_harmonic_analysis('CPI', 'Yield_10Y', fs=12)
dmd_results = engine.dynamic_modes_analysis(n_modes=5)
regimes = engine.regime_detection(threshold=0.3)

# Step 2: Generate all visualizations
create_full_prism_report(
    panel_df=panel,
    vector_results=vector_results,
    harmonic_results=harmonic_results,
    dmd_results=dmd_results,
    regimes=regimes,
    col_a='CPI',
    col_b='Yield_10Y'
)

# OR generate individual plots:
# VectorVisualizations.plot_vector_trajectory_3d(panel)
# HarmonicVisualizations.plot_phase_space(panel, 'CPI', 'Yield_10Y')
# DMDVisualizations.plot_eigenvalue_spectrum(dmd_results)
# RegimeVisualizations.plot_regime_timeline(regimes, panel)
    """)
    print("=" * 70)


# ============================================================================
# BONUS: ADDITIONAL SPECIALIZED VISUALIZATIONS
# ============================================================================

class AdvancedVCFVisualizations:
    """
    Additional specialized plots for deep analysis
    """
    
    @staticmethod
    def plot_correlation_evolution(panel_df: pd.DataFrame, window: int = 50):
        """
        Rolling correlation heatmap over time
        """
        from scipy.stats import pearsonr
        
        cols = panel_df.columns
        n_cols = len(cols)
        n_windows = len(panel_df) - window + 1
        
        # Compute rolling correlations
        corr_series = np.zeros((n_cols, n_cols, n_windows))
        
        for t in range(n_windows):
            window_data = panel_df.iloc[t:t+window]
            for i in range(n_cols):
                for j in range(n_cols):
                    if i == j:
                        corr_series[i, j, t] = 1.0
                    else:
                        corr, _ = pearsonr(window_data.iloc[:, i], window_data.iloc[:, j])
                        corr_series[i, j, t] = corr
        
        # Plot heatmap for selected pairs
        fig, axes = plt.subplots(n_cols-1, 1, figsize=(14, 4*n_cols))
        if n_cols == 2:
            axes = [axes]
        
        for idx in range(n_cols-1):
            ax = axes[idx]
            im = ax.imshow(corr_series[idx, idx+1:, :], aspect='auto', cmap='RdBu_r', 
                          vmin=-1, vmax=1, interpolation='bilinear')
            ax.set_ylabel(f'{cols[idx]} vs Others', fontsize=10, fontweight='bold')
            ax.set_yticks(range(n_cols-idx-1))
            ax.set_yticklabels(cols[idx+1:], fontsize=9)
            plt.colorbar(im, ax=ax, label='Correlation')
        
        axes[-1].set_xlabel('Time Window', fontsize=11, fontweight='bold')
        fig.suptitle('Rolling Correlation Evolution', fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_phase_coherence_histogram(panel_df: pd.DataFrame, col_a: str, col_b: str):
        """
        Distribution of phase differences
        """
        signal_a = panel_df[col_a].values
        signal_b = panel_df[col_b].values
        
        phase_a = np.angle(hilbert(signal_a))
        phase_b = np.angle(hilbert(signal_b))
        
        phase_diff = np.angle(np.exp(1j * (phase_b - phase_a)))  # Wrapped difference
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(phase_diff, bins=50, color='purple', alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='In-Phase')
        ax1.axvline(np.pi, color='blue', linestyle='--', linewidth=2, label='Anti-Phase')
        ax1.axvline(-np.pi, color='blue', linestyle='--', linewidth=2)
        
        ax1.set_xlabel('Phase Difference (radians)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax1.set_title(f'Phase Difference Distribution: {col_a} vs {col_b}', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Polar histogram
        ax2 = plt.subplot(122, projection='polar')
        bins_polar = np.linspace(-np.pi, np.pi, 25)
        counts, _ = np.histogram(phase_diff, bins=bins_polar)
        
        theta = (bins_polar[:-1] + bins_polar[1:]) / 2
        width = 2 * np.pi / len(counts)
        
        bars = ax2.bar(theta, counts, width=width, alpha=0.7, edgecolor='black')
        
        # Color by quadrant
        for bar, th in zip(bars, theta):
            if -np.pi/4 <= th < np.pi/4:
                bar.set_facecolor('green')  # In-phase
            elif np.pi*3/4 <= th or th < -np.pi*3/4:
                bar.set_facecolor('red')  # Anti-phase
            else:
                bar.set_facecolor('yellow')  # Quadrature
        
        ax2.set_title('Polar Phase Distribution', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_wavelet_coherence_heatmap(panel_df: pd.DataFrame, col_a: str, col_b: str):
        """
        Time-frequency coherence map using wavelets
        """
        import pywt
        
        signal_a = panel_df[col_a].values
        signal_b = panel_df[col_b].values
        
        scales = np.arange(1, min(64, len(signal_a)//4))
        
        # Continuous wavelet transform
        coef_a, freqs_a = pywt.cwt(signal_a, scales, 'morl')
        coef_b, freqs_b = pywt.cwt(signal_b, scales, 'morl')
        
        # Cross-wavelet power
        cross_power = coef_a * np.conj(coef_b)
        
        # Wavelet coherence
        coherence = np.abs(cross_power)**2 / (np.abs(coef_a)**2 * np.abs(coef_b)**2 + 1e-10)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.contourf(range(len(signal_a)), scales, coherence, levels=20, cmap='RdYlBu_r')
        
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Scale (Period)', fontsize=11, fontweight='bold')
        ax.set_title(f'Wavelet Coherence: {col_a} vs {col_b}', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coherence', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_granger_causality_network(panel_df: pd.DataFrame, max_lag: int = 12):
        """
        Network diagram showing Granger causality relationships
        """
        from scipy.stats import f
        
        cols = panel_df.columns
        n = len(cols)
        
        # Compute Granger causality (simplified F-test)
        causality_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Simplified Granger test
                    x = panel_df.iloc[:, i].values
                    y = panel_df.iloc[:, j].values
                    
                    # Lagged regression (very simplified)
                    X_lagged = np.column_stack([np.roll(x, k) for k in range(1, max_lag+1)])
                    X_lagged = X_lagged[max_lag:]
                    y_target = y[max_lag:]
                    
                    # Correlation as proxy for causality strength
                    corr = np.abs(np.corrcoef(X_lagged.T, y_target)[:-1, -1].mean())
                    causality_matrix[i, j] = corr
        
        # Plot network
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Position nodes in circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Draw edges (causality arrows)
        for i in range(n):
            for j in range(n):
                if i != j and causality_matrix[i, j] > 0.3:  # Threshold
                    strength = causality_matrix[i, j]
                    ax.arrow(x_pos[i], y_pos[i], 
                           (x_pos[j] - x_pos[i]) * 0.8, 
                           (y_pos[j] - y_pos[i]) * 0.8,
                           head_width=0.1, head_length=0.1, 
                           fc='blue', ec='blue', alpha=strength, linewidth=2*strength)
        
        # Draw nodes
        ax.scatter(x_pos, y_pos, s=1000, c='lightcoral', edgecolors='black', 
                  linewidths=3, zorder=10)
        
        # Labels
        for i, (x, y, col) in enumerate(zip(x_pos, y_pos, cols)):
            ax.text(x*1.2, y*1.2, col, fontsize=12, fontweight='bold',
                   ha='center', va='center')
        
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Granger Causality Network (Simplified)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiscale_decomposition(panel_df: pd.DataFrame, col: str):
        """
        Visualize EMD decomposition into intrinsic modes
        """
        from scipy.signal import hilbert
        
        signal = panel_df[col].values
        
        # Simplified EMD (using frequency bands as proxy)
        from scipy.signal import butter, filtfilt
        
        # Decompose into 4 scales
        scales = [
            ('High Freq', (0.3, 0.5)),
            ('Medium Freq', (0.1, 0.3)),
            ('Low Freq', (0.01, 0.1)),
            ('Trend', (0, 0.01))
        ]
        
        fig, axes = plt.subplots(len(scales) + 1, 1, figsize=(14, 10), sharex=True)
        
        # Original
        axes[0].plot(panel_df.index, signal, linewidth=2, color='black')
        axes[0].set_ylabel('Original', fontsize=10, fontweight='bold')
        axes[0].set_title(f'{col} - Multi-Scale Decomposition', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Decomposed scales
        colors = ['red', 'orange', 'blue', 'green']
        for idx, ((name, (low, high)), color) in enumerate(zip(scales, colors)):
            if high == 0.5:
                b, a = butter(3, high, btype='low')
            elif low == 0:
                b, a = butter(3, high, btype='low')
            else:
                b, a = butter(3, [low, high], btype='band')
            
            filtered = filtfilt(b, a, signal)
            
            axes[idx+1].plot(panel_df.index, filtered, linewidth=1.5, color=color, alpha=0.8)
            axes[idx+1].set_ylabel(name, fontsize=10, fontweight='bold')
            axes[idx+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


# Add to master function
def create_advanced_prism_report(panel_df: pd.DataFrame, col_a: str, col_b: str):
    """
    Generate advanced specialized visualizations
    """
    print("\n" + "=" * 70)
    print("ADVANCED VCF VISUALIZATIONS")
    print("=" * 70)
    
    print("\n1. Correlation Evolution")
    AdvancedVCFVisualizations.plot_correlation_evolution(panel_df, window=50)
    
    print(f"\n2. Phase Coherence Distribution: {col_a} vs {col_b}")
    AdvancedVCFVisualizations.plot_phase_coherence_histogram(panel_df, col_a, col_b)
    
    print(f"\n3. Wavelet Coherence Heatmap: {col_a} vs {col_b}")
    AdvancedVCFVisualizations.plot_wavelet_coherence_heatmap(panel_df, col_a, col_b)
    
    print("\n4. Granger Causality Network")
    AdvancedVCFVisualizations.plot_granger_causality_network(panel_df, max_lag=12)
    
    print(f"\n5. Multi-Scale Decomposition: {col_a}")
    AdvancedVCFVisualizations.plot_multiscale_decomposition(panel_df, col_a)
    
    print("\n" + "=" * 70)
    print("ADVANCED REPORT COMPLETE")
    print("=" * 70)