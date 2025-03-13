import numpy as np
import xarray as xr
import pandas as pd

# Load the datasets
landcover_ds = xr.open_dataset('LUH_Historical_1850_2014.nc')
reforestation_ds = xr.open_dataset('cook_reforestation_Canesm.nc')

landcover_ds['frac']



# Convert numeric time values to string and then to datetime
# time_as_strings = [str(int(time_value)) for time_value in landcover_ds.time.values]
# landcover_ds['time'] = pd.to_datetime(time_as_strings, format='%Y%m%d')

# Find the index for 2014 and extract the data for this year
#year_2014_idx = np.where(landcover_ds.time.dt.year == 2014)[0][0]
data_2014 = landcover_ds.isel(time=-1)

# Create a new time coordinate from 2014 to 2200 using the correct frequency code 'Y'
new_years = pd.date_range(start='2014', end='2201', freq='YE')  # Yearly frequency, corrected

# Initialize the new dataset
new_ds = xr.Dataset({
    'frac': (('time', 'lev', 'lat', 'lon'), np.repeat(data_2014.frac.values[None, :, :, :], len(new_years), axis=0)),
    'lon': ('lon', data_2014.lon.values),  # Correct use of .values for numpy array extraction
    'lat': ('lat', data_2014.lat.values),
    'lev': ('lev', data_2014.lev.values),
    'time': ('time', new_years)
})

# Print to verify the structure and content of the new dataset
# print(new_ds)

# Define forest, crop, and grass level identifiers in 'lev' dimension
forest_levels = [1, 2, 3, 4, 5]  # Forest
crop_levels = [6, 7]             # Crops
grass_levels = [8, 9]            # Grass

# Adjust for zero-based indexing if necessary
forest_indices = [x - 1 for x in forest_levels]
crop_indices = [x - 1 for x in crop_levels]
grass_indices = [x - 1 for x in grass_levels]

# Your reforestation map and adjustments come here
# Define the ratios for increasing forests and
# Use the initial (2014) fractions for forests, crops, and grasslands
initial_forest_fraction = new_ds['frac'].isel(time=0).sel(lev=forest_levels).sum(dim='lev')
initial_crop_fraction = new_ds['frac'].isel(time=0).sel(lev=crop_levels).sum(dim='lev')
initial_grass_fraction = new_ds['frac'].isel(time=0).sel(lev=grass_levels).sum(dim='lev') #data_2014.frac.sel(lev=forest_indices).sum(dim='lev')

print(new_ds['frac'])

# Calculate the total initial fractions for normalization
total_initial_fraction = initial_forest_fraction + initial_crop_fraction + initial_grass_fraction

# Apply changes year by year
for year_index in range(55):
    # Calculate the fraction of the reforestation map to apply for this year
    expansion_factor = (year_index + 1) / 55.0
    yearly_reforestation_impact = reforestation_ds['Band1'] * expansion_factor 
    correct_time = new_years[year_index]
    # Proportionally increase forest cover based on the reforestation fraction
    forest_increase = yearly_reforestation_impact.expand_dims({'lev':len(forest_levels), 'time':1})
    forest_increase['lev'] = forest_levels  # Assign the correct 'lev' coordinates
    forest_increase['time'] = [new_years[year_index]]  # Assign the correct 'time' coordinate
    forest_increase_aligned =forest_increase.reindex_like(new_ds['frac'], method='nearest')#, new_ds['frac'], join='left', fill_value=0)
    
    for fl in forest_levels:
        forest_increase= yearly_reforestation_impact * (initial_forest_fraction / total_initial_fraction) # ensure correct level index
        
        #forest_increase_aligned, _ =xr.align(forest_increase, new_ds['frac'], join='left', fill_value=0)
        correct_time = new_years[year_index]
        new_ds['frac'].loc[dict(time=correct_time, lev=fl)] += forest_increase_aligned.sel(time=correct_time, lev=fl)  # Adjust indexing for 0-based
        

    for cl in crop_levels:
        crop_decrease = (yearly_reforestation_impact * (initial_crop_fraction / total_initial_fraction)).expand_dims({'lev':1, 'time':1})

        crop_decrease['lev'] = [cl] # ensure correct level index
        crop_decrease['time']=[correct_time]
        crop_decrease_aligned = crop_decrease.reindex_like( new_ds['frac'], method='nearest', fill_value=0)
        temp_result = new_ds['frac'].sel(time=correct_time, lev=cl) - crop_decrease_aligned.sel(time=correct_time, lev=cl)
        new_ds['frac'].loc[dict(time=correct_time, lev=cl)] = temp_result
    
    for gl in grass_levels:
        grass_reduction = (yearly_reforestation_impact * (initial_grass_fraction / total_initial_fraction)).expand_dims({'lev':1, 'time':1})  # ensure correct level index
        grass_reduction['lev'] = [gl]
        grass_reduction['time'] = [correct_time]
        grass_reduction_aligned = grass_reduction.reindex_like(new_ds['frac'], method='nearest', fill_value = 0)
        temp_result2 = new_ds['frac'].sel(time=correct_time, lev=gl) - grass_reduction_aligned.sel(time=correct_time, lev=gl)    
        new_ds['frac'].loc[dict(time=correct_time, lev=gl)] = temp_result2   # Adjust indexing for 0-based

    # Ensure no negative values
   # new_ds['frac'].loc[dict(time=year_index, lev=gl)] = new_ds['frac'].loc[dict(time=year_index, lev=gl)].clip(min=0)

    # Normalize each year to ensure fractions sum to 1 or less
    total_frac = new_ds['frac'].sel(time=correct_time).sum(dim='lev')
    normalization_factor = 1 / total_frac
    new_ds['frac'].loc[dict(time=correct_time)] *= normalization_factor

# After 33 years, keep the values constant for each subsequent year
final_state = new_ds['frac'].sel(time=new_years[54]).copy()
for year_index in range(55, len(new_years)):
    new_ds['frac'].loc[dict(time=new_years[year_index])] = final_state

# Save the dataset
new_ds.to_netcdf('reforestation_projection_2014_to_2200.nc')


