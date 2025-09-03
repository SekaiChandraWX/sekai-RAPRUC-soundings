# 🌪️ RAP/RUC Atmospheric Sounding Plotter

A comprehensive Streamlit web application for generating **RAP (Rapid Refresh)** and **RUC (Rapid Update Cycle)** atmospheric sounding plots with detailed meteorological analysis.

## 🌟 Features

- **Comprehensive Sounding Analysis**: Skew-T log-P diagrams with temperature, dewpoint, and wind profiles
- **Interactive Hodographs**: Wind patterns by altitude with storm motion vectors
- **Advanced Parameters**: CAPE/CIN, wind shear, storm-relative helicity, and composite indices
- **Hemisphere-Specific Calculations**: Automatic detection with appropriate storm motion selection
- **Hazard Assessment**: Automated severe weather potential evaluation
- **Professional Dark Theme**: Matching NCEP operational products
- **High-Quality Output**: Publication-ready plots at 150 DPI
- **Historical Coverage**: Seamless transition between RUC (2005-2012) and RAP (2012-present) models

## 🗺️ Coverage & Data

- **Geographic Coverage**: Continental United States and Southern Canada
- **Temporal Coverage**: 2005 to near real-time (~6-12 hours delay)
- **Temporal Resolution**: Hourly (00Z-23Z UTC)
- **Spatial Resolution**: 13 km grid spacing (RAP), 20 km (RUC legacy)
- **Data Source**: NOAA National Centers for Environmental Information (NCEI)
- **Model Evolution**:
  - **RUC (2005-2012)**: Rapid Update Cycle
  - **RAP (2012-present)**: Rapid Refresh

## 📋 Requirements

- Python 3.8 or higher
- Internet connection for GRIB file downloads
- ~50-100 MB temporary storage per sounding
- All dependencies listed in `requirements.txt`

## 🚀 Installation

1. **Clone or download the repository**:
   ```bash
   git clone <your-repo-url>
   cd rap-sounding-plotter
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the provided URL (typically `http://localhost:8501`)

## 💻 Usage

### Location Input
- **City/Place Name**: Enter any North American location (e.g., "Oklahoma City", "Denver", "Toronto")
- **Coordinates**: Manually input latitude/longitude for precise locations

### Date & Time Selection
- **Date Range**: 2005-01-01 to ~12 hours ago
- **Time Options**: All 24 hours (00Z-23Z) available
- **UTC Time**: All times are in Coordinated Universal Time
- **Historical Coverage**: Automatic model selection (RUC vs RAP) based on date

### Analysis Output
The application generates comprehensive atmospheric analysis including:

#### Thermodynamic Parameters
- Surface-Based CAPE (SBCAPE)
- Mixed Layer CAPE (MLCAPE)  
- Most Unstable CAPE (MUCAPE)
- Convective Inhibition (CIN)
- Lifted Condensation Level (LCL)
- 3km CAPE (low-level instability)

#### Kinematic Parameters
- Bulk wind shear (0-1km, 0-3km, 0-6km, 0-8km)
- Storm-Relative Helicity (SRH)
- Effective bulk wind difference
- Storm motion vectors (Bunkers method)

#### Composite Indices
- Supercell Tornado Parameter (STP)
- Critical angle calculations
- Automated hazard assessment

## 🌐 Deployment

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment
Compatible with:
- **Streamlit Community Cloud**
- **Heroku** (with appropriate buildpacks)
- **AWS/GCP/Azure** container services
- **Docker** deployment

### Environment Variables
No API keys or special configuration required - uses public NCEI data archives.

## 🔧 Technical Details

### Data Access
- **Source**: NOAA NCEI Historical Weather Data Archives
- **Format**: GRIB/GRIB2 binary files
- **Variables**: Temperature, humidity, winds, geopotential height
- **Pressure Levels**: 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 275, 250, 225, 200, 175, 150, 125, 100 hPa

### Model Transitions
The app automatically selects the appropriate model and data source:

1. **RUC 252 Grid** (2005-2008): 20km resolution, Lambert Conformal
2. **RUC 130 Grid** (2008-2012): 13km resolution, improved physics
3. **RAP Historical** (2012-2020): 13km resolution, advanced data assimilation
4. **RAP Current** (2020-present): Latest operational version

### Processing Pipeline
1. **Date Validation**: Ensures requested time is within available data range
2. **Model Selection**: Automatically chooses RUC vs RAP based on date
3. **GRIB Download**: Downloads compressed meteorological data (~5-15 MB)
4. **Data Extraction**: Uses pygrib to decode isobaric level data
5. **Grid Interpolation**: Finds nearest model grid point to requested location
6. **Quality Control**: Filters invalid/missing data points
7. **Calculations**: Derives all meteorological parameters
8. **Visualization**: Generates comprehensive 4-panel analysis plot
9. **Cleanup**: Removes temporary files to conserve disk space

### Performance
- **Typical Processing Time**: 45-90 seconds
- **Data Volume**: ~5-15 MB per sounding (GRIB compressed)
- **Memory Usage**: ~200-400 MB during processing
- **Concurrent Users**: Supports multiple simultaneous requests

## 🎨 Plot Components

### 1. Skew-T Log-P Diagram
- Temperature and dewpoint profiles
- Wind barbs at pressure levels
- Parcel trajectories and stability areas
- Reference lines (dry/moist adiabats, mixing ratios)

### 2. Hodograph
- Wind vectors colored by height layer
- Storm motion vectors (right/left-moving)
- Critical angle visualization
- Hemisphere-appropriate calculations

### 3. Parameter Tables
- Three-column layout with calculated indices
- Thermodynamic, kinematic, and composite parameters
- Storm-relative calculations

### 4. Hazard Assessment
- Automated severe weather categorization
- Color-coded threat levels
- Based on operational forecasting criteria

## ⚠️ Limitations

- **Geographic**: North America only (RAP/RUC domain restriction)
- **Temporal**: ~6-12 hour delay for most recent data
- **File Size**: Temporary GRIB downloads require adequate disk space
- **Resolution**: 13km grid may not capture microscale features
- **Historical Gaps**: Some data may be missing during model transitions

## 📊 Model Specifications

### RAP (Rapid Refresh) - 2012-Present
- **Resolution**: 13 km horizontal, 50 vertical levels
- **Domain**: North America (including southern Canada, northern Mexico)
- **Physics**: Advanced microphysics, land surface model
- **Data Assimilation**: 3D-VAR with cloud analysis
- **Forecast Length**: 21 hours (analysis + forecast)

### RUC (Rapid Update Cycle) - 2005-2012
- **Resolution**: 13-20 km horizontal, 50 vertical levels
- **Domain**: Continental United States focus
- **Physics**: Multi-layer soil model, explicit convection
- **Data Assimilation**: 3D-VAR analysis
- **Legacy Status**: Replaced by RAP in 2012

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional composite parameters (EHI, BRN, etc.)
- Export functionality (CSV, JSON, SHARPpy format)
- Batch processing capabilities
- Mobile-responsive design improvements
- Historical climatology comparisons

## 📄 License

This project is open source. RAP/RUC data courtesy of NOAA NCEI.

## 🙏 Acknowledgments

- **NOAA NCEI**: RAP/RUC data archives and distribution
- **MetPy Development Team**: Meteorological calculations library
- **UCAR/Unidata**: GRIB decoding tools and standards
- **NOAA EMC**: RAP/RUC model development and operations

## 📞 Support

For technical issues:
1. Check the console for detailed error messages
2. Verify location is within North America
3. Ensure selected date/time has available data
4. Try a different time period if download fails
5. Check internet connection for GRIB file access

### Common Issues
- **Download timeouts**: NCEI servers may be slow during peak usage
- **Missing data**: Some historical periods may have gaps
- **Large files**: RAP GRIB files can be 15+ MB for recent data

---

*Created by @Sekai_WX - Professional atmospheric analysis tools for research and operations*