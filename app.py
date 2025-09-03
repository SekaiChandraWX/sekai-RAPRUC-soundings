import streamlit as st
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import tempfile
from geopy.geocoders import Nominatim
import time
import gc
from threading import Lock
import warnings
import urllib.request
import re
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from PIL import Image

warnings.filterwarnings("ignore")

# Import pygrib with error handling
try:
    import pygrib
    PYGRIB_AVAILABLE = True
except ImportError:
    PYGRIB_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="RAP/RUC Sounding Plotter", 
    page_icon="🌪️",
    layout="wide"
)

# Resource management
processing_lock = Lock()

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    try:
        import pygrib
    except ImportError:
        missing.append("pygrib")
    
    try:
        import metpy
    except ImportError:
        missing.append("metpy")
        
    try:
        from geopy.geocoders import Nominatim
    except ImportError:
        missing.append("geopy")
    
    return len(missing) == 0, missing

def get_coordinates(location_str):
    """Geocoding with RAP/RUC domain validation"""
    try:
        geolocator = Nominatim(user_agent="rap_sounding_viewer", timeout=5)
        time.sleep(1)
        location = geolocator.geocode(location_str, timeout=5)
        if location:
            lat, lon = location.latitude, location.longitude
            
            # RAP/RUC domain check (North America focus)
            # Rough bounds: 15°N to 65°N, 140°W to 60°W
            if not (15.0 <= lat <= 65.0) or not (-140.0 <= lon <= -60.0):
                st.error(f"❌ Location outside RAP/RUC domain (North America)")
                return None, None
                
            return lat, lon
        return None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None

def get_model_info(case_date):
    """Determine model type and URLs based on date"""
    ruc_change = datetime(2008, 10, 29, 23)
    rap_change = datetime(2012, 5, 1, 12)
    new_rap_change = datetime(2020, 5, 17, 0)
    
    date1 = case_date.strftime('%Y%m')
    date2 = case_date.strftime('%Y%m%d')
    date3 = case_date.strftime('%y%m%d/%H%S')
    
    if case_date < ruc_change:
        grib_file = f'ruc2anl_252_{case_date.strftime("%Y%m%d_%H%S")}_000.grb'
        model_name = "RUC 252"
        snd_date = f'RUC   {date3}'
    elif (case_date >= ruc_change) and (case_date < rap_change):
        grib_file = f'ruc2anl_130_{case_date.strftime("%Y%m%d_%H%S")}_000.grb2'
        model_name = "RUC 130"
        snd_date = f'RUC   {date3}'
    elif (case_date >= rap_change) and (case_date < new_rap_change):
        grib_file = f'rap_130_{case_date.strftime("%Y%m%d_%H%S")}_000.grb2'
        model_name = "RAP Historical"
        snd_date = f'RAP   {date3}'
    else:
        grib_file = f'rap_130_{case_date.strftime("%Y%m%d_%H%S")}_000.grb2'
        model_name = "RAP Current"
        snd_date = f'RAP   {date3}'
    
    if case_date < new_rap_change:
        url = f'https://www.ncei.noaa.gov/data/rapid-refresh/access/historical/analysis/{date1}/{date2}/{grib_file}'
    else:
        url = f'https://www.ncei.noaa.gov/data/rapid-refresh/access/rap-130-13km/analysis/{date1}/{date2}/{grib_file}'
    
    return grib_file, url, model_name, snd_date

def download_rap_data(case_date, progress_bar):
    """Download RAP/RUC GRIB file from NCEI"""
    grib_file, url, model_name, snd_date = get_model_info(case_date)
    
    progress_bar.progress(10, f"Downloading {model_name} data...")
    
    try:
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, grib_file)
        
        # Download with progress indication
        progress_bar.progress(25, f"Downloading {grib_file}...")
        urllib.request.urlretrieve(url, local_path)
        
        progress_bar.progress(50, f"{model_name} data downloaded successfully!")
        
        return local_path, model_name, snd_date
        
    except Exception as e:
        raise Exception(f"Download failed from {url}: {str(e)}")

def decode_grib_data(grib_path, case_date, lat, lon, progress_bar):
    """Decode GRIB data and extract atmospheric profile"""
    progress_bar.progress(60, "Decoding GRIB data...")
    
    try:
        grbs = pygrib.open(grib_path)
        grbs.seek(0)
        
        # Extract variables from GRIB file
        z_grb = grbs.select(name="Geopotential height", typeOfLevel='isobaricInhPa')
        t_grb = grbs.select(name="Temperature", typeOfLevel='isobaricInhPa')
        rh_grb = grbs.select(name="Relative humidity", typeOfLevel='isobaricInhPa')
        u_grb = grbs.select(name="U component of wind", typeOfLevel='isobaricInhPa')
        v_grb = grbs.select(name="V component of wind", typeOfLevel='isobaricInhPa')
        
        # Find nearest grid point
        lats, lons = z_grb[0].latlons()
        idx = np.unravel_index(np.sqrt((lats - lat)**2 + (lons - lon)**2).argmin(), lats.shape)
        new_lat, new_lon = lats[idx], lons[idx]
        
        # Extract data based on model version (different ordering)
        date_rapv3 = datetime(2016, 8, 23, 12)
        
        if case_date < date_rapv3:
            z = np.array([z_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for z_grb in z_grb])
            t = np.array([t_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for t_grb in t_grb])
            rh = np.array([rh_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for rh_grb in rh_grb])
            u = np.array([u_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for u_grb in u_grb])
            v = np.array([v_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for v_grb in v_grb])
        else:
            z = np.array([z_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for z_grb in reversed(z_grb)])
            t = np.array([t_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for t_grb in reversed(t_grb)])
            rh = np.array([rh_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for rh_grb in reversed(rh_grb)])
            u = np.array([u_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for u_grb in reversed(u_grb)])
            v = np.array([v_grb.data(new_lat, new_lat, new_lon, new_lon)[0][0] for v_grb in reversed(v_grb)])
        
        grbs.close()
        
        # Add units and convert
        z = z * units("m")
        t = t * units("degK") 
        rh = rh * units("%")
        u = u * units("m/s")
        v = v * units("m/s")
        
        # Generate pressure levels
        pres = np.arange(1000.0, 75.0, -25) * units.hPa
        hght = z
        temp = t.to('degC')
        dwpt = mpcalc.dewpoint_from_relative_humidity(t, rh).to('degC')
        wdir = mpcalc.wind_direction(u, v)
        wspd = mpcalc.wind_speed(u, v).to('kt')
        
        return {
            'pressure': pres,
            'height': hght, 
            'temperature': temp,
            'dewpoint': dwpt,
            'wind_direction': wdir,
            'wind_speed': wspd,
            'u_wind': u,
            'v_wind': v,
            'actual_lat': new_lat,
            'actual_lon': new_lon
        }
        
    except Exception as e:
        raise Exception(f"GRIB decoding failed: {str(e)}")

def create_comprehensive_rap_plot(data, lat, location, model_name, case_date, progress_bar):
    """Create comprehensive RAP/RUC sounding plot"""
    progress_bar.progress(85, "Creating comprehensive sounding plot...")
    
    try:
        plot_filename = f"rap_sounding_{int(time.time())}.jpg"
        temp_plot_path = os.path.join(tempfile.gettempdir(), plot_filename)
        
        # Create title string
        title_str = f"{model_name} Sounding for {location} on {case_date.strftime('%Y-%m-%d')} at {case_date.strftime('%H')}Z"
        
        create_rap_plot_comprehensive(data, temp_plot_path, title_str, lat)
        
        # Read the image file and return as matplotlib figure for Streamlit
        img = Image.open(temp_plot_path)
        
        # Convert PIL image to matplotlib figure
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.imshow(img)
        ax.axis('off')
        
        # Clean up temp file
        if os.path.exists(temp_plot_path):
            os.remove(temp_plot_path)
            
        return fig
        
    except Exception as e:
        raise Exception(f"Comprehensive plot creation failed: {str(e)}")

def create_rap_plot_comprehensive(data, file_path, title_str, lat=None):
    """Create comprehensive RAP/RUC sounding plot - adapted from original script"""
    try:
        # Extract data with units
        pres = data['pressure']
        hght = data['height']
        temp = data['temperature']
        dwpt = data['dewpoint']
        wdir = data['wind_direction']
        wspd = data['wind_speed']
        
        # Convert RAP data to the format our plotting logic expects
        pressure = np.array(pres.magnitude)
        height = np.array(hght.to('meter').magnitude)
        temperature = np.array(temp.to('degC').magnitude)
        dewpoint = np.array(dwpt.to('degC').magnitude)
        u_wind, v_wind = mpcalc.wind_components(wspd, wdir)
        u_wind = np.array(u_wind.to('knot').magnitude)
        v_wind = np.array(v_wind.to('knot').magnitude)
        wspd_vals = np.array(wspd.to('knot').magnitude)
        wdir_vals = np.array(wdir.magnitude)
        
        # Sort by decreasing pressure (surface to top)
        sort_idx = np.argsort(pressure)[::-1]
        pressure = pressure[sort_idx]
        height = height[sort_idx]
        temperature = temperature[sort_idx]
        dewpoint = dewpoint[sort_idx]
        u_wind = u_wind[sort_idx]
        v_wind = v_wind[sort_idx]
        wspd_vals = wspd_vals[sort_idx]
        wdir_vals = wdir_vals[sort_idx]
        
        # Filter valid data
        valid_mask = (
            (pressure > 0) &
            (np.isfinite(pressure)) & (np.isfinite(temperature)) & (np.isfinite(dewpoint)) &
            (temperature > -100) & (temperature < 60) &
            (dewpoint > -100) & (dewpoint < 60) &
            (dewpoint <= temperature + 0.1)
        )
        
        # Apply mask
        p = pressure[valid_mask] * units.hPa
        T = temperature[valid_mask] * units.degC
        Td = dewpoint[valid_mask] * units.degC
        height_m = height[valid_mask] * units.meter
        u = u_wind[valid_mask] * units.knot
        v = v_wind[valid_mask] * units.knot
        
        # Calculate wind components for hodograph
        wdir_rad = np.deg2rad(wdir_vals[valid_mask])
        u_calc = -wspd_vals[valid_mask] * np.sin(wdir_rad) * units.knot
        v_calc = -wspd_vals[valid_mask] * np.cos(wdir_rad) * units.knot
        
        # Determine hemisphere - use provided lat or assume Northern Hemisphere
        is_southern_hemisphere = lat is not None and lat < 0
        
        # Calculate basic parameters with error handling
        try:
            # Surface-based parcel
            sb_parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0])
            sb_cape, sb_cin = mpcalc.cape_cin(p, T, Td, sb_parcel_prof)
            
            # Mixed layer parcel
            ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
            ml_parcel_prof = mpcalc.parcel_profile(p, ml_t, ml_td)
            ml_cape, ml_cin = mpcalc.mixed_layer_cape_cin(p, T, ml_parcel_prof, depth=50 * units.hPa)
            
            # Most unstable parcel
            mu_cape, mu_cin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)
            
            # LCL
            lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
            
            # LCL height estimation
            new_p = np.append(p[p > lcl_p], lcl_p)
            new_t = np.append(T[p > lcl_p], lcl_t)
            lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)
        except:
            sb_cape = ml_cape = mu_cape = 0 * units('J/kg')
            sb_cin = ml_cin = mu_cin = 0 * units('J/kg')
            lcl_p, lcl_t = p[0], T[0]
            lcl_height = 1000 * units.meter
        
        try:
            # Storm motion - both RM and LM
            storm_motion = mpcalc.bunkers_storm_motion(p, u_calc, v_calc, height_m)
            storm_u_rm, storm_v_rm = storm_motion[0]
            storm_u_lm, storm_v_lm = storm_motion[1]
            
            # Hemisphere-specific storm motion selection
            if is_southern_hemisphere:
                primary_storm_u, primary_storm_v = storm_u_lm, storm_v_lm
                storm_type_label = "LM"
                line_color = 'red'
            else:
                primary_storm_u, primary_storm_v = storm_u_rm, storm_v_rm
                storm_type_label = "RM"
                line_color = 'green'
            
            # Shear calculations
            shear_1km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=1 * units.km))
            shear_3km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=3 * units.km))
            shear_6km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=6 * units.km))
            shear_8km = mpcalc.wind_speed(*mpcalc.bulk_shear(p, u_calc, v_calc, height=height_m, depth=8 * units.km))
            
            # SRH calculations - hemisphere-specific
            srh_1km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=1 * units.km,
                                                     storm_u=primary_storm_u, storm_v=primary_storm_v)[2]
            srh_3km = mpcalc.storm_relative_helicity(height_m, u_calc, v_calc, depth=3 * units.km,
                                                     storm_u=primary_storm_u, storm_v=primary_storm_v)[2]
            
            # Use absolute values for Southern Hemisphere
            if is_southern_hemisphere:
                srh_1km = abs(srh_1km)
                srh_3km = abs(srh_3km)
            
            # Effective layer calculations (simplified)
            esrh = srh_3km
            ebwd = shear_6km
            
            # STP calculations
            try:
                # STPF (Thompson et al. 2003)
                stpf = (sb_cape / (1500 * units('J/kg'))) * \
                       (srh_1km / (150 * units('m^2/s^2'))) * \
                       (shear_6km / (20 * units('m/s'))) * \
                       ((2000 * units.meter - lcl_height) / (1000 * units.meter))
                stpf = stpf.to('dimensionless').magnitude
                
                # STPE (Thompson et al. 2012)
                stpe = (ml_cape / (1500 * units('J/kg'))) * \
                       (esrh / (150 * units('m^2/s^2'))) * \
                       (ebwd / (12 * units('m/s'))) * \
                       ((2000 * units.meter - lcl_height) / (1000 * units.meter)) * \
                       ((ml_cin + 200 * units('J/kg')) / (150 * units('J/kg')))
                stpe = stpe.to('dimensionless').magnitude
            except:
                stpf = stpe = 0
            
            # Critical angle calculation
            try:
                critical_angle = mpcalc.critical_angle(p, u_calc, v_calc, height_m, primary_storm_u, primary_storm_v)
            except:
                critical_angle = 0 * units.degrees
        
        except:
            storm_u_rm = storm_v_rm = storm_u_lm = storm_v_lm = 0 * units.knot
            primary_storm_u = primary_storm_v = 0 * units.knot
            shear_1km = shear_3km = shear_6km = shear_8km = 0 * units('m/s')
            srh_1km = srh_3km = esrh = 0 * units('m^2/s^2')
            ebwd = 0 * units('m/s')
            stpf = stpe = 0
            critical_angle = 0 * units.degrees
            storm_type_label = "RM"
            line_color = 'green'
        
        # Calculate 3CAPE
        try:
            surface_height = height_m[0]
            km3_height = surface_height + 3000 * units.meter
            km3_mask = height_m <= km3_height
            if np.any(km3_mask):
                p_3km = p[km3_mask]
                T_3km = T[km3_mask]
                Td_3km = Td[km3_mask]
                sb_parcel_3km = mpcalc.parcel_profile(p_3km, T[0], Td[0])
                cape_3km, _ = mpcalc.cape_cin(p_3km, T_3km, Td_3km, sb_parcel_3km)
            else:
                cape_3km = 0 * units('J/kg')
        except:
            cape_3km = 0 * units('J/kg')
        
        # Lapse rate calculations
        try:
            sfc_height = height_m[0]
            km1_idx = np.argmin(np.abs(height_m - (sfc_height + 1000 * units.meter)))
            km3_idx = np.argmin(np.abs(height_m - (sfc_height + 3000 * units.meter)))
            lapse_1km = (T[0] - T[km1_idx]) / (1 * units.km)
            lapse_3km = (T[0] - T[km3_idx]) / (3 * units.km)
        except:
            lapse_1km = lapse_3km = 6.5 * units('K/km')
        
        # === PLOTTING SECTION - DARK THEME ===
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 12), facecolor='#2F2F2F')
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2.5, 1])
        
        # TOP LEFT: Skew-T plot
        skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
        skew.ax.set_facecolor('#2F2F2F')
        
        # Plot the data
        skew.plot(p, T, 'r', linewidth=2.5)
        skew.plot(p, Td, 'g', linewidth=2.5)
        skew.plot_barbs(p, u_calc, v_calc, barbcolor='white', flagcolor='white')
        
        # Set limits and labels
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)
        skew.ax.set_xlabel(f'Temperature ({T.units:~P})', color='white')
        skew.ax.set_ylabel(f'Pressure ({p.units:~P})', color='white')
        skew.ax.tick_params(colors='white')
        
        # Plot LCL
        skew.plot(lcl_p, lcl_t, 'wo', markerfacecolor='white', markeredgecolor='black', markersize=8)
        
        # Plot parcel profile
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
            skew.plot(p, prof, color='#FFFF99', linewidth=2, linestyle='--')
            skew.shade_cin(p, T, prof, Td, alpha=0.4)
            skew.shade_cape(p, T, prof, alpha=0.4)
        except:
            pass
        
        # Add reference lines
        skew.ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.7)
        skew.plot_dry_adiabats(colors='white', alpha=0.3)
        skew.plot_moist_adiabats(colors='white', alpha=0.3)
        skew.plot_mixing_lines(colors='white', alpha=0.3)
        skew.ax.set_title('Skew-T Log-P Diagram', fontsize=14, fontweight='bold', color='white')
        skew.ax.grid(True, alpha=0.3, color='white')
        
        # TOP RIGHT: Hodograph
        hodo_ax = fig.add_subplot(gs[0, 1], facecolor='#2F2F2F')
        h = Hodograph(hodo_ax, component_range=80)
        h.add_grid(increment=20, color='white', alpha=0.3)
        
        # Color-code hodograph by height layers
        try:
            height_agl = height_m - height_m[0]
            layers = [
                (0, 3000, 'red', '0-3km'),
                (3000, 6000, 'green', '3-6km'),
                (6000, 9000, 'yellow', '6-9km'),
                (9000, 20000, 'lightblue', '>9km')
            ]
            
            for bottom, top, color, label in layers:
                height_values = height_agl.to('meter').magnitude
                mask = (height_values >= bottom) & (height_values < top)
                if np.any(mask):
                    h.plot(u_calc[mask], v_calc[mask], color=color, linewidth=3, label=label)
            
            # Add both RM and LM storm motion vectors
            h.ax.plot(storm_u_rm.magnitude, storm_v_rm.magnitude, 'go', markersize=8,
                      markerfacecolor='green', markeredgecolor='white', linewidth=2, label='RM Storm Motion')
            h.ax.plot(storm_u_lm.magnitude, storm_v_lm.magnitude, 'ro', markersize=8,
                      markerfacecolor='red', markeredgecolor='white', linewidth=2, label='LM Storm Motion')
            
            # Add critical angle line to hemisphere-appropriate storm motion
            h.ax.plot([0, primary_storm_u.magnitude], [0, primary_storm_v.magnitude],
                      '--', color=line_color, alpha=0.8, linewidth=2)
        except:
            pass
        
        hemisphere_text = "SHEM" if is_southern_hemisphere else "NHEM"
        hodo_ax.set_title(f'Hodograph ({hemisphere_text} - {storm_type_label} Primary)',
                          fontsize=14, fontweight='bold', color='white')
        hodo_ax.tick_params(colors='white')
        
        # Add critical angle text
        try:
            angle_value = critical_angle.magnitude
            hodo_ax.text(0.02, 0.02, f'Critical Angle ({storm_type_label}): {angle_value:.0f} degrees',
                         transform=hodo_ax.transAxes, fontsize=11, fontweight='bold',
                         color='white', bbox=dict(boxstyle="round,pad=0.3",
                                                  facecolor='#2F2F2F', alpha=0.9, edgecolor='white'))
        except:
            pass
        
        # BOTTOM LEFT: Parameters
        params_ax = fig.add_subplot(gs[1, 0], facecolor='#2F2F2F')
        params_ax.axis('off')
        
        base_col1 = 0.02 + 0.125
        base_col2 = 0.30 + 0.125
        base_col3 = 0.58 + 0.125
        
        thermodynamic_params = [
            f"SBCAPE: {sb_cape:~.0f}",
            f"MLCAPE: {ml_cape:~.0f}",
            f"MUCAPE: {mu_cape:~.0f}",
            f"SBCIN: {sb_cin:~.0f}",
            f"MLCIN: {ml_cin:~.0f}",
            f"3CAPE: {cape_3km:~.0f}",
            f"LCL Height: {lcl_height:~.0f}"
        ]
        
        kinematic_params = [
            f"0-1km Shear: {shear_1km:~.1f}",
            f"0-3km Shear: {shear_3km:~.1f}",
            f"0-6km Shear: {shear_6km:~.1f}",
            f"0-8km Shear: {shear_8km:~.1f}",
            f"Eff Bulk Shear: {ebwd:~.1f}",
            f"0-1km SRH ({storm_type_label}): {srh_1km:~.0f}",
            f"0-3km SRH ({storm_type_label}): {srh_3km:~.0f}",
            f"Eff SRH ({storm_type_label}): {esrh:~.0f}"
        ]
        
        composite_params = [
            f"STPF: {stpf:.2f}",
            f"STPE: {stpe:.2f}",
            f"RM Storm U: {storm_u_rm:~.1f}",
            f"RM Storm V: {storm_v_rm:~.1f}",
            f"LM Storm U: {storm_u_lm:~.1f}",
            f"LM Storm V: {storm_v_lm:~.1f}",
            f"0-1km Lapse: {lapse_1km:~.1f}"
        ]
        
        # Titles
        title_y = 0.95
        params_ax.text(base_col1, title_y, "THERMODYNAMIC PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold',
                       fontfamily='monospace', color='white')
        params_ax.text(base_col2, title_y, "KINEMATIC PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold',
                       fontfamily='monospace', color='white')
        params_ax.text(base_col3, title_y, "COMPOSITE PARAMETERS:",
                       transform=params_ax.transAxes, fontsize=11, fontweight='bold',
                       fontfamily='monospace', color='white')
        
        # Parameters
        param_start_y = 0.85
        for i, text in enumerate(thermodynamic_params):
            params_ax.text(base_col1, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')
        
        for i, text in enumerate(kinematic_params):
            params_ax.text(base_col2, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')
        
        for i, text in enumerate(composite_params):
            params_ax.text(base_col3, param_start_y - i * 0.1, text, transform=params_ax.transAxes,
                           fontsize=10, fontfamily='monospace', color='white')
        
        params_ax.set_title('Calculated Parameters', fontweight='bold', fontsize=14, color='white')
        
        # BOTTOM RIGHT: Hazard Assessment
        hazard_ax = fig.add_subplot(gs[1, 1], facecolor='#2F2F2F')
        hazard_ax.axis('off')
        
        try:
            cape_val = sb_cape.to('J/kg').magnitude
            srh_3km_val = abs(srh_3km.to('m^2/s^2').magnitude)
            shear_6km_val = shear_6km.to('m/s').magnitude
            
            if (stpf >= 8 or stpe >= 10 or
                    (stpf >= 4 and stpe >= 6 and srh_3km_val >= 400 and shear_6km_val >= 25) or
                    (srh_3km_val >= 500 and shear_6km_val >= 30 and cape_val >= 2000)):
                hazard_type = "PDS TOR"
                hazard_color = "#C85A8A"
            elif (stpf >= 3 or stpe >= 4 or
                  (srh_3km_val >= 300 and shear_6km_val >= 20 and cape_val >= 1500) or
                  (stpf >= 2 and stpe >= 3 and srh_3km_val >= 200)):
                hazard_type = "TOR"
                hazard_color = "red"
            elif (stpf >= 1 or stpe >= 1.5 or
                  (srh_3km_val >= 150 and shear_6km_val >= 15 and cape_val >= 1000)):
                hazard_type = "MRGL TOR"
                hazard_color = "darkred"
            elif (shear_6km_val >= 15 and cape_val >= 1000):
                hazard_type = "SVR"
                hazard_color = "orange"
            elif (shear_6km_val >= 10 and cape_val >= 500):
                hazard_type = "MRGL SVR"
                hazard_color = "yellow"
            elif (cape_val >= 2500 and shear_6km_val < 10):
                hazard_type = "FLASH FLOOD"
                hazard_color = "blue"
            elif cape_val < 100:
                hazard_type = "NONE"
                hazard_color = "green"
            else:
                hazard_type = "SVR"
                hazard_color = "orange"
        except:
            hazard_type = "UNKNOWN"
            hazard_color = "gray"
        
        # Hazard display
        hazard_ax.text(0.5, 0.80, "PREDICTED", transform=hazard_ax.transAxes,
                       fontsize=16, fontweight='bold', ha='center', color='white')
        hazard_ax.text(0.5, 0.65, "HAZARD TYPE", transform=hazard_ax.transAxes,
                       fontsize=16, fontweight='bold', ha='center', color='white')
        hazard_ax.text(0.5, 0.40, hazard_type, transform=hazard_ax.transAxes,
                       fontsize=22, fontweight='bold', ha='center', color=hazard_color,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='#2F2F2F',
                                 edgecolor=hazard_color, linewidth=3))
        
        # Title and attribution
        fig.suptitle(f'Comprehensive Sounding Analysis - {title_str}',
                     fontsize=16, fontweight='bold', color='white')
        plt.figtext(0.5, 0.02, 'Plotted by @Sekai_WX (RAP/RUC Data)',
                    ha='center', fontsize=8, style='italic', color='white')
        
        # Save plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, left=0.00, right=0.95, hspace=0.3, wspace=0.2)
        plt.savefig(file_path, format='jpg', dpi=150, bbox_inches='tight', facecolor='#2F2F2F')
        plt.close()
        
    except Exception as e:
        print(f"Enhanced plotting error: {e}")
        raise Exception(f"Comprehensive plot creation failed: {str(e)}")

def process_rap_sounding(date_input, hour, lat, lon, location):
    """Main processing function for RAP/RUC sounding"""
    
    if not processing_lock.acquire(blocking=False):
        raise Exception("Another sounding request is processing. Please wait and try again.")
    
    try:
        case_date = datetime(date_input.year, date_input.month, date_input.day, hour)
        
        progress_bar = st.progress(0, "Initializing RAP/RUC sounding request...")
        
        # Download RAP/RUC data
        grib_path, model_name, snd_date = download_rap_data(case_date, progress_bar)
        
        try:
            # Decode GRIB data
            data = decode_grib_data(grib_path, case_date, lat, lon, progress_bar)
            
            # Create comprehensive plot
            fig = create_comprehensive_rap_plot(data, lat, location, model_name, case_date, progress_bar)
            
            progress_bar.progress(100, "Complete!")
            time.sleep(0.5)
            progress_bar.empty()
            
            return fig, data, model_name
            
        finally:
            # Clean up GRIB file
            if os.path.exists(grib_path):
                try:
                    os.remove(grib_path)
                except:
                    pass  # Don't fail if cleanup fails
            gc.collect()
            
    finally:
        processing_lock.release()

# Streamlit UI
st.title("🌪️ RAP/RUC Atmospheric Sounding Plotter")
st.markdown("### Rapid Refresh & Rapid Update Cycle - Comprehensive Analysis")

# Check dependencies
deps_available, missing = check_dependencies()
if not deps_available:
    st.error(f"❌ Required libraries are not installed: {', '.join(missing)}")
    st.info("This tool requires: pygrib, metpy, geopy, and matplotlib")
    st.stop()

st.markdown("""
Generate **RAP (Rapid Refresh)** and **RUC (Rapid Update Cycle)** sounding plots for North America. 
High-resolution atmospheric analysis with hourly temporal resolution from 2005 to near real-time.
""")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📅 Date & Time")
    
    # Date input (RAP/RUC available from 2005 with ~6-12 hour delay)
    today = datetime.now().date()
    max_date = today - timedelta(hours=12)  # RAP/RUC delay
    min_date = datetime(2005, 1, 1).date()  # RAP/RUC start
    
    date_input = st.date_input(
        "Date", 
        value=max_date,  # ✅ Fixed
        min_value=min_date, 
        max_value=max_date,  # ✅ Fixed (also changed max_date to max_value)
        help="RAP/RUC data available from 2005 to ~12 hours ago"
    )
    
    hour_input = st.selectbox(
        "Hour (UTC)", 
        options=list(range(24)), 
        index=12,
        format_func=lambda x: f"{x:02d}:00",
        help="RAP/RUC provides hourly data (00-23 UTC)"
    )
    
    # Show model selection info
    if date_input:
        test_date = datetime.combine(date_input, datetime.min.time().replace(hour=hour_input))
        _, _, model_name, _ = get_model_info(test_date)
        st.info(f"📡 **Model**: {model_name}")
    
    st.subheader("🌍 Location")
    
    # Add domain warning
    st.info("📍 **RAP/RUC Coverage:** North America (US, Southern Canada, Northern Mexico)")
    
    location_method = st.radio(
        "Location Input Method",
        ["City/Place Name", "Coordinates (Lat, Lon)"]
    )
    
    if location_method == "City/Place Name":
        location_input = st.text_input(
            "Enter North American location", 
            placeholder="e.g., Oklahoma City, Denver, Toronto, Mexico City"
        )
        lat, lon = None, None
        if location_input:
            with st.spinner("Finding coordinates..."):
                lat, lon = get_coordinates(location_input)
                if lat and lon:
                    hemisphere = "Southern" if lat < 0 else "Northern"
                    st.success(f"📍 {lat:.4f}°, {lon:.4f}° ({hemisphere} Hemisphere)")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude", min_value=15.0, max_value=65.0, 
                                value=39.0, step=0.1, help="RAP/RUC: 15°N to 65°N")
        with col_lon:
            lon = st.number_input("Longitude", min_value=-140.0, max_value=-60.0, 
                                value=-98.0, step=0.1, help="RAP/RUC: 140°W to 60°W")
        
        if lat is not None:
            hemisphere = "Southern" if lat < 0 else "Northern"
            st.info(f"📍 {lat:.4f}°, {abs(lon):.4f}°W ({hemisphere} Hemisphere)")
    
    # Processing time warning
    st.info("⏱️ **Processing Time:** 15-30 seconds")
    
    generate_button = st.button("🚀 Generate RAP/RUC Sounding Analysis", type="primary")

with col2:
    st.subheader("📊 Comprehensive RAP/RUC Atmospheric Sounding")
    
    if generate_button:
        if lat is not None and lon is not None:
            try:
                location_display = location_input if location_method == "City/Place Name" else f"{lat:.2f}°N, {abs(lon):.2f}°W"
                fig, data, model_name = process_rap_sounding(date_input, hour_input, lat, lon, location_display)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                gc.collect()
                
                st.success("✅ Comprehensive RAP/RUC sounding analysis complete!")
                st.info("💡 Right-click on the plot to save it to your device as JPG.")
                
                # Additional info
                case_date = datetime.combine(date_input, datetime.min.time().replace(hour=hour_input))
                actual_lat = data['actual_lat']
                actual_lon = data['actual_lon']
                distance_km = np.sqrt((lat - actual_lat)**2 + (lon - actual_lon)**2) * 111  # rough km conversion
                
                st.caption(f"📅 Analysis time: {case_date.strftime('%Y-%m-%d %H:%M UTC')}")
                st.caption(f"🎯 Model: {model_name}")
                st.caption(f"📍 Nearest grid point: {actual_lat:.3f}°, {abs(actual_lon):.3f}°W (±{distance_km:.0f} km)")
                
            except Exception as e:
                st.error(f"❌ Error generating sounding: {str(e)}")
                if "outside RAP/RUC domain" in str(e):
                    st.info("💡 RAP/RUC covers North America only. Try a location within the US, Canada, or northern Mexico.")
                elif "Download failed" in str(e):
                    st.info("💡 Data may not be available for this date/time. Try a more recent time or check your internet connection.")
                elif "Another request" in str(e):
                    st.info("💡 Only one user can process at a time. Please wait and retry.")
        else:
            st.warning("⚠️ Please provide a valid location within North America.")

# Information sections
with st.expander("🌪️ About RAP/RUC Atmospheric Soundings"):
    st.markdown("""
    **RAP (Rapid Refresh)** and **RUC (Rapid Update Cycle)** are high-resolution numerical weather 
    prediction models developed by NOAA for short-term forecasting and analysis over North America.
    
    **Model Evolution:**
    - **RUC (2005-2012)**: Rapid Update Cycle - predecessor to RAP
    - **RAP (2012-present)**: Rapid Refresh - current operational model
    
    **Key Features:**
    - **Spatial Resolution**: 13 km grid spacing (RAP), 20 km (legacy RUC)
    - **Temporal Resolution**: Hourly analysis and forecasts
    - **Coverage**: Continental US, southern Canada, northern Mexico
    - **Vertical Levels**: 50 model levels from surface to 10 mb
    - **Data Assimilation**: 3D-VAR with extensive observational inputs
    
    **Comprehensive Analysis Includes:**
    - **Skew-T log-P diagrams** with temperature, dewpoint, and wind profiles
    - **Hodographs** showing wind patterns by altitude with storm motion vectors
    - **CAPE/CIN calculations** for convective potential assessment
    - **Wind shear analysis** at multiple levels (0-1km, 0-3km, 0-6km, 0-8km)
    - **Storm-Relative Helicity (SRH)** with hemisphere-specific calculations
    - **Supercell Tornado Parameter (STP)** for severe weather potential
    - **Automated hazard assessment** (TOR, SVR, MRGL, etc.)
    - **Professional dark theme** matching operational products
    
    **Advantages over Global Models:**
    - Higher resolution captures mesoscale features
    - Rapid update cycle (hourly vs 6-hourly)
    - Optimized for North American weather patterns
    - Extensive use of surface and upper-air observations
    """)

with st.expander("📚 Model Specifications & Data Sources"):
    st.markdown("""
    **RAP Model (2012-Present):**
    - **Grid**: 13 km Lambert Conformal Conic
    - **Domain**: 1799 x 1059 grid points
    - **Levels**: 50 hybrid levels (surface to 10 mb)
    - **Physics**: WSM6 microphysics, Grell-Freitas convection
    - **Land Surface**: Noah LSM with 4 soil layers
    - **Boundary Conditions**: GFS model
    - **Data Assimilation**: GSI 3D-VAR with cloud analysis
    
    **RUC Model (2005-2012):**
    - **Grid**: 13-20 km Lambert Conformal
    - **Domain**: Variable based on version
    - **Physics**: Mixed-phase microphysics
    - **Land Surface**: RUC LSM with 6-9 soil levels
    - **Boundary Conditions**: GFS/NAM models
    
    **Data Archive Access:**
    - **Current RAP**: NCEI Rapid Refresh archive
    - **Historical RAP**: NCEI historical weather data
    - **RUC Legacy**: NCEI historical archives
    - **Format**: GRIB/GRIB2 compressed binary
    - **Availability**: ~6-12 hour delay for recent data
    
    **Observational Inputs:**
    - Radiosondes (twice daily)
    - Surface weather stations (hourly)
    - Aircraft data (ACARS/AMDAR)
    - Wind profilers and RASS
    - Satellite-derived winds and temperatures
    - Radar reflectivity and radial velocity
    """)

with st.expander("🔧 Technical Processing Details"):
    st.markdown("""
    **File Processing:**
    - **Download**: GRIB/GRIB2 files from NCEI (5-15 MB each)
    - **Decoding**: pygrib library for binary data extraction
    - **Interpolation**: Nearest neighbor grid point selection
    - **Quality Control**: Invalid data filtering and consistency checks
    
    **Atmospheric Variables:**
    - **Temperature**: Celsius conversion from Kelvin
    - **Humidity**: Relative humidity to dewpoint conversion
    - **Winds**: U/V components to speed/direction
    - **Heights**: Geopotential to geometric height
    - **Pressure**: Standard isobaric levels (1000-100 hPa)
    
    **Calculation Methods:**
    - **CAPE/CIN**: MetPy parcel theory calculations
    - **Shear**: Bulk wind difference over specified layers
    - **SRH**: Storm-relative helicity using Bunkers storm motion
    - **Composite Indices**: Thompson et al. STP formulations
    - **Lapse Rates**: Environmental vs. adiabatic temperature profiles
    
    **Performance Optimization:**
    - **Concurrent Processing**: Thread-safe download/processing
    - **Memory Management**: Automatic cleanup of temporary files
    - **Error Handling**: Graceful degradation for missing data
    - **Caching**: Temporary file management to reduce redundant downloads
    """)

st.markdown("---")
st.markdown("*High-resolution atmospheric analysis using RAP/RUC data • Created by @Sekai_WX*")