"""Figure generation: diagnostic plots, interpolated PSF maps and summary overlays."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def plot_catalog_diagnostics(results_csv, image_path, individuals_dir):
    """Plots catalog-vs-instrumental magnitudes and the calibrated magnitude histogram."""
    if os.path.exists(results_csv):
        csv_df = pd.read_csv(results_csv)
        if 'mag_abs' in csv_df.columns and 'mag_instr' in csv_df.columns:
            valid_mags = csv_df.dropna(subset=['mag_abs', 'mag_instr'])
            if not valid_mags.empty:
                plt.figure(figsize=(10, 6))
                
                plt.scatter(valid_mags['mag_instr'], valid_mags['mag_abs'], c='blue', s=10, alpha=0.5, label='All Extracted Stars')
                
                if 'catalog_mag' in csv_df.columns:
                    matched_stars = csv_df.dropna(subset=['catalog_mag', 'mag_instr'])
                else:
                    matched_stars = pd.DataFrame()
                    
                if not matched_stars.empty:
                    plt.scatter(matched_stars['mag_instr'], matched_stars['catalog_mag'], c='orange', s=10, alpha=0.9, label='Catalog Matched Stars')
                    residuals = matched_stars['mag_abs'] - matched_stars['catalog_mag']
                    fit_error = residuals.std()
                    match_count = len(matched_stars)
                else:
                    fit_error = np.nan
                    match_count = 0
                
                # Plot the theoretical fit line
                x_range = np.linspace(valid_mags['mag_instr'].min(), valid_mags['mag_instr'].max(), 100)
                zp_calc = valid_mags['mag_abs'].iloc[0] - valid_mags['mag_instr'].iloc[0]
                plt.plot(x_range, x_range + zp_calc, 'r--', alpha=0.8, label=f'Fit (ZP={zp_calc:.2f})')
                
                plt.xlabel("Instrumental Magnitude")
                plt.ylabel("Magnitude")
                plt.title(f"Absolute Photometry Overview - {os.path.basename(image_path)}")
                leg = plt.legend(loc='lower right')
                if hasattr(leg, 'legend_handles'):
                    for lh in leg.legend_handles:
                        lh.set_alpha(1.0)
                elif hasattr(leg, 'legendHandles'):
                    for lh in leg.legendHandles:
                        lh.set_alpha(1.0)
                plt.grid(True, alpha=0.3)
                
                if match_count > 0:
                    stats_text = f"Matched Stars: {match_count}\nFit Error (StdDev): {fit_error:.3f} mag"
                    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                
                mag_plot_name = os.path.join(individuals_dir, f'{os.path.basename(image_path)}_mag_comparison.png')
                plt.savefig(mag_plot_name, bbox_inches='tight')
                plt.close()
                
                # Also generate the abs_mag_hist plot since it's in the README
                plt.figure(figsize=(10, 6))
                plt.hist(valid_mags['mag_abs'], bins=30, color='purple', alpha=0.7, edgecolor='black')
                plt.xlabel("Absolute Magnitude")
                plt.ylabel("Count")
                plt.title(f"Absolute Magnitude Distribution - {os.path.basename(image_path)}")
                plt.grid(axis='y', alpha=0.3)
                hist_plot_name = os.path.join(individuals_dir, f'{os.path.basename(image_path)}_abs_mag_hist.png')
                plt.savefig(hist_plot_name, bbox_inches='tight')
                plt.close()


def plot_detected(data, df, image_path, individuals_dir):
    """Plots the star field with detections circled."""
    # Simple star field plot
    plt.figure(figsize=(10, 8))
    plt.imshow(data, origin='lower', cmap='gray', vmax=np.percentile(data, 99))
    plt.scatter(df['xcentroid'], df['ycentroid'], s=20, edgecolor='red', facecolor='none', alpha=0.5)
    plt.title(f"Detected Stars - {os.path.basename(image_path)}")
    plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_detected.png'), bbox_inches='tight')
    plt.close()


def plot_psf_maps(df, wcs, nx, ny, image_path, individuals_dir):
    """Plots interpolated FWHM, PSF-size, orientation and elongation maps."""
    # Interpolated Maps
    if len(df) > 10:
        print("Generating interpolated maps...")
        grid_x, grid_y = np.mgrid[0:nx:100j, 0:ny:100j]
        
        # FWHM Map
        mean_fwhm = np.nanmean(df['fwhm'])
        std_fwhm = np.nanstd(df['fwhm'])
        fwhm_vmin = mean_fwhm - (2 * std_fwhm)
        fwhm_vmax = mean_fwhm + (2 * std_fwhm)
        
        plt.figure(figsize=(10, 8))
        grid_fwhm = griddata((df['xcentroid'], df['ycentroid']), df['fwhm'], (grid_x, grid_y), method='cubic')
        ax = plt.gca()
        im = ax.imshow(grid_fwhm.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis', vmin=fwhm_vmin, vmax=fwhm_vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='FWHM (pixels)')
        ax.set_title(f"FWHM Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_fwhm_map.png'), bbox_inches='tight')
        plt.close()

        if wcs:
            print("Generating PSF arcsec map (vectorized)...")
            # df['fwhm_arc'] is already populated by compute_fwhm_arcsec() upstream.
            mean_arc = np.nanmean(df['fwhm_arc'])
            std_arc = np.nanstd(df['fwhm_arc'])
            arc_vmin = mean_arc - (2 * std_arc)
            arc_vmax = mean_arc + (2 * std_arc)
            
            plt.figure(figsize=(10, 8))
            grid_fwhm_arc = griddata((df['xcentroid'], df['ycentroid']), df['fwhm_arc'], (grid_x, grid_y), method='cubic')
            ax = plt.gca()
            im = ax.imshow(grid_fwhm_arc.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis', vmin=arc_vmin, vmax=arc_vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='FWHM (arcsec)')
            ax.set_title(f"PSF Size Map (arcsec) - {os.path.basename(image_path)}")
            plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_psf_arcsec_map.png'), bbox_inches='tight')
            plt.close()

        # Theta Map
        plt.figure(figsize=(10, 8))
        grid_theta = griddata((df['xcentroid'], df['ycentroid']), df['theta'], (grid_x, grid_y), method='cubic')
        ax = plt.gca()
        im = ax.imshow(grid_theta.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Theta (degrees)')
        ax.set_title(f"PSF Orientation Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_theta_map.png'), bbox_inches='tight')
        plt.close()

        # Elongation Map
        plt.figure(figsize=(10, 8))
        grid_elongation = griddata((df['xcentroid'], df['ycentroid']), df['elongation'], (grid_x, grid_y), method='cubic')
        ax = plt.gca()
        im = ax.imshow(grid_elongation.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Elongation')
        ax.set_title(f"PSF Elongation Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_elongation_map.png'), bbox_inches='tight')
        plt.close()


def plot_distortion_map(wcs, nx, ny, image_path, individuals_dir):
    """Plots the astrometric distortion map, returning the sampled pixel-scale grid."""
    scales = None
    if wcs:
        print("Generating distortion map...")
        # Create a grid of points for WCS sampling
        gy, gx = np.mgrid[0:ny:20j, 0:nx:20j]
        scales = np.zeros_like(gx)
        
        # Calculate reference scale at the image center (zero distortion reference)
        cx, cy = nx / 2, ny / 2
        s1_c = wcs.pixel_to_world(cx, cy)
        s2_c = wcs.pixel_to_world(cx + 1, cy)
        s3_c = wcs.pixel_to_world(cx, cy + 1)
        center_scale_x = s1_c.separation(s2_c).arcsec
        center_scale_y = s1_c.separation(s3_c).arcsec
        center_scale_avg = np.sqrt(center_scale_x * center_scale_y)
        
        # Vectorized grid sampling
        all_gx = gx.flatten()
        all_gy = gy.flatten()
        
        sky1_all = wcs.pixel_to_world(all_gx, all_gy)
        sky2_all = wcs.pixel_to_world(all_gx + 1, all_gy)
        sky3_all = wcs.pixel_to_world(all_gx, all_gy + 1)
        
        scale_x = sky1_all.separation(sky2_all).arcsec
        scale_y = sky1_all.separation(sky3_all).arcsec
        scales = np.sqrt(scale_x * scale_y).reshape(gx.shape)
        
        # Calculate the gradient of the scale field
        grad_y, grad_x = np.gradient(scales)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = np.where(grad_mag == 0, 1e-10, grad_mag)
        dir_x = -grad_x / grad_mag
        dir_y = -grad_y / grad_mag
        dist_mag = np.abs(scales - center_scale_avg)
        u_raw = dist_mag * dir_x
        v_raw = dist_mag * dir_y
        
        raw_mag = np.sqrt(u_raw**2 + v_raw**2)
        max_raw = np.max(raw_mag)
        
        if max_raw > 0:
            u_comp = (u_raw / max_raw) * 75
            v_comp = (v_raw / max_raw) * 75
            key_length_px = 50
            key_length_arcsec = (50.0 / 75.0) * max_raw
        else:
            u_comp = u_raw
            v_comp = v_raw
            key_length_arcsec = 0.0
            key_length_px = 50 
        
        # Interpolate heatmap
        grid_x_f, grid_y_f = np.mgrid[0:nx:100j, 0:ny:100j]
        grid_dist = griddata((gx.flatten(), gy.flatten()), scales.flatten(), 
                             (grid_x_f, grid_y_f), method='cubic')
        
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        im = ax.imshow(grid_dist.T, extent=(0, nx, 0, ny), origin='lower', cmap='viridis')
        q = ax.quiver(gx, gy, u_comp, v_comp, color='white', alpha=0.8, 
                      scale_units='xy', angles='xy', scale=1.0,
                      headwidth=2.25, headlength=3.75, headaxislength=3.375)
        ax.quiverkey(q, 0.9, 1.05, 50, f'{key_length_arcsec:.3f}"/pix relative to center', 
                     labelpos='E', coordinates='axes', color='black')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Local Pixel Scale (arcsec/pixel)')
        ax.set_title(f"Astrometric Distortion Map - {os.path.basename(image_path)}")
        plt.savefig(os.path.join(individuals_dir, f'{os.path.basename(image_path)}_distortion_map.png'), bbox_inches='tight')
        plt.close()
    return scales


def plot_summary(data, df, df_outliers, wcs, scales, nx, ny, mean, std, zero_point,
                 image_path, figures_dir):
    """Plots the master summary overlay with detection circles and a statistics box."""
    # Summary Figure
    print("Generating summary overlay figure...")
    plt.figure(figsize=(12, 10))
    plt.imshow(data, origin='lower', cmap='gray', vmin=np.percentile(data, 0.02), vmax=np.percentile(data, 99))

    # Draw circles around stars: radius = 2 * FWHM_pixels
    for _, row in df.iterrows():
        circ = plt.Circle((row['xcentroid'], row['ycentroid']), 2 * row['fwhm'],
                          color='yellow', fill=False, linewidth=0.8, alpha=0.6)
        plt.gca().add_patch(circ)

    # Draw red circles around outlier stars (disabled per user request)
    # if not df_outliers.empty:
    #     for _, row in df_outliers.iterrows():
    #         circ = plt.Circle((row['xcentroid'], row['ycentroid']), 2 * row['fwhm'],
    #                           color='red', fill=False, linewidth=0.8, alpha=0.6)
    #         plt.gca().add_patch(circ)

    # Info box
    fwhm_px = df['fwhm'].values
    fwhm_arc = df['fwhm_arc'].values
    def fmt_stats(px, arc, func):
        v_px = func(px)
        v_arc = func(arc)
        if np.isnan(v_arc):
            return f"{v_px:.2f} px (N/A)"
        return f"{v_px:.2f} px ({v_arc:.2f}\")"

    if wcs:
        if scales is not None:
            avg_scale = np.mean(scales)
        else:
            try:
                sky1_ref = wcs.pixel_to_world(nx/2, ny/2)
                sky2_ref = wcs.pixel_to_world(nx/2 + 1, ny/2)
                sky3_ref = wcs.pixel_to_world(nx/2, ny/2 + 1)
                avg_scale = np.sqrt(sky1_ref.separation(sky2_ref).arcsec * sky1_ref.separation(sky3_ref).arcsec)
            except: avg_scale = -999.0
    else: avg_scale = -999.0

    info_text = (
        f"Image: {os.path.basename(image_path)}\n"
        f"Size: {nx} x {ny} px\n"
        f"Scale: {avg_scale:.3f}\"/pix\n"
        f"Zero Point: {zero_point if zero_point else 'N/A'}\n"
        f"Stars Found: {len(df) + len(df_outliers)} (Valid: {len(df)})\n"
        f"Background: {mean:.2f} (StdDev: {std:.2f})\n\n"
        f"FWHM:\n"
        f"  Min: {fmt_stats(fwhm_px, fwhm_arc, np.min)}\n"
        f"  Max: {fmt_stats(fwhm_px, fwhm_arc, np.max)}\n"
        f"  Avg: {fmt_stats(fwhm_px, fwhm_arc, np.mean)}\n"
    )
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10, fontfamily='monospace')
    
    #plt.title(f"Star Detection Summary - {os.path.basename(image_path)}")
    plt.savefig(os.path.join(figures_dir, f'{os.path.basename(image_path)}_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_composite(image_list, csvs_dir, figures_dir):
    """Plots combined magnitude comparison and histogram figures across a batch."""
    print("Generating composite magnitude plot...")
    all_dfs = []
    for img in image_list:
        csv_path = os.path.join(csvs_dir, f'{os.path.basename(img)}_results.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'mag_instr' in df.columns and 'mag_abs' in df.columns:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        valid_df = combined_df.dropna(subset=['mag_instr', 'mag_abs'])
        
        if not valid_df.empty:
            plt.figure(figsize=(10, 6))
            
            plt.scatter(valid_df['mag_instr'], valid_df['mag_abs'], alpha=0.2, s=5, c='blue', label='All Measured Stars')

            if 'catalog_mag' in combined_df.columns:
                matched_stars = combined_df.dropna(subset=['catalog_mag', 'mag_instr'])
            else:
                matched_stars = pd.DataFrame()

            if not matched_stars.empty:
                plt.scatter(matched_stars['mag_instr'], matched_stars['catalog_mag'], alpha=0.1, s=5, c='orange', label='Catalog Matched Stars')
                residuals = matched_stars['mag_abs'] - matched_stars['catalog_mag']
                fit_error = residuals.std()
                match_count = len(matched_stars)
            else:
                fit_error = np.nan
                match_count = 0
            
            median_zp = (valid_df['mag_abs'] - valid_df['mag_instr']).median()
            x_range = np.linspace(valid_df['mag_instr'].min(), valid_df['mag_instr'].max(), 100)
            plt.plot(x_range, x_range + median_zp, 'r--', alpha=0.8, label=f'Fit (Median ZP={median_zp:.2f})')
            
            plt.xlabel("Instrumental Magnitude")
            plt.ylabel("Absolute Magnitude")
            # plt.title("Composite Absolute Photometry - All Images")
            leg = plt.legend(loc='lower right')
            if hasattr(leg, 'legend_handles'):
                for lh in leg.legend_handles:
                    lh.set_alpha(1.0)
            elif hasattr(leg, 'legendHandles'):
                for lh in leg.legendHandles:
                    lh.set_alpha(1.0)
            plt.grid(True, alpha=0.3)
            
            if match_count > 0:
                stats_text = f"Total Matched: {match_count}\nMedian ZP: {median_zp:.3f}\nFit Error (StdDev): {fit_error:.3f} mag"
            else:
                stats_text = f"Total Measured: {len(valid_df)}\nMedian ZP: {median_zp:.3f}"
                
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            composite_path = os.path.join(figures_dir, "composite_mag_comparison.png")
            plt.savefig(composite_path, bbox_inches='tight')
            plt.close()
            print(f"Composite plot saved to {composite_path}")

            # Composite Histogram
            plt.figure(figsize=(10, 6))
            plt.hist(valid_df['mag_abs'], bins=40, color='purple', alpha=0.7, edgecolor='black')
            plt.xlabel("Absolute Magnitude")
            plt.ylabel("Frequency")
            # plt.title("Composite Absolute Magnitude Distribution - All Images")
            plt.grid(True, alpha=0.3)
            
            # Add stats box for the histogram
            histo_stats = (
                f"Count: {len(valid_df)}\n"
                f"Min: {valid_df['mag_abs'].min():.2f}\n"
                f"Max: {valid_df['mag_abs'].max():.2f}\n"
                f"Median: {valid_df['mag_abs'].median():.2f}"
            )
            plt.text(0.95, 0.95, histo_stats, transform=plt.gca().transAxes, 
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            composite_hist_path = os.path.join(figures_dir, "composite_mag_histogram.png")
            plt.savefig(composite_hist_path, bbox_inches='tight')
            plt.close()
            print(f"Composite histogram saved to {composite_hist_path}")
        else:
            print("No valid magnitude data found across images for composite plot.")
