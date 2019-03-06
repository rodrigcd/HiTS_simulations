    
# Image Sequence Simulator for ZTF and HiTS

Python code designed to simulate light curves and sequence of images for a certain observation plan and telescope. In this case it is for the HiTS Survey (link), but it will be generalized for other surveys.

## Instructions, inputs and outputs of each step

The code is currently divided in three steps. Get the observation plan, observation conditions and camera parameters, light curve sampling and image sequence generation. In order to use each part, run the following:

### Observation plan, observation conditions and camera parameters

Getting real data from HiTS images. This part is not needed if you build your own camera and obs condition file. The data is not in this repository.

```
# It will read real supernovae images and get supernovae stamps sequence observation 
# conditions and camera parameters, generating a pickle file "sn_data.pkl"
python DataFromHits.py 
sn_data.keys() = ["HiTS01SN", "HiTS02SN", ...]
sn_data["HiTS01SN"].keys() = ['images', 'diff', 'headers', 'psf']
sn_data["HiTS01SN"]["images"].shape = (21, 21, 26) = (x_size, y_size, epochs) # Same format for 'diff' and 'psf'
sn_data["HiTS01SN"]["headers"].keys() = ['obs_days', 'sky_brightness', 'sky_sigma', 'airmass', 'ccd_num', 
                                         'exp_time', 'gain', 'seeing', 'pixel_scale', 'read_noise', 'saturation']
```
The final product of this part is camera_and_obs_cond.pkl, you could build your oun observation plan using a different survey as long as you keep the format of this file. In this case we do the following:

```
python generate_obs_conditions.py # Generate camera and obs condition file using HiTS data
camera_obs_cond.keys() = ['camera_params', 'obs_conditions', 'psf']

# Camera
camera_obs_cond["camera_params"].keys() = ['CCD25', 'CCD52', 'CCD86', 'CCD54', ...]
camera_obs_cond["camera_params"]["CCD25"] = {'ccd_num': 25,
 					     'gain': 4.104, #[e-/ADU]
 					     'read_noise': 6.119064, #[e-]
 					     'saturation': 40469.0, #[ADU]
					     'pixel_scale': 0.27, #[arcsec/pixel]
 					     'zp_g': 25.399156, #camera zero point not used
 					     'zp_i': 25.313254,
 					     'zp_r': 25.474396,
 					     'zp_u': 23.546145}

# Obs conditions per field
camera_obs_cond["obs_conditions"].keys() = ['Field01', 'Field02', 'Field04', ...]
type(camera_obs_cond["obs_conditions"]["Field01"]) = list # one element per epoch
camera_obs_cond["obs_conditions"]["Field01"][0] = {'airmass': 1.6,
  						   'epoch': 1.0,
  						   'exp_time': 86.0,
  						   'filter': 'r',
  						   'limmag3': 23.8178227406003, #[magnitude]
  						   'limmag5': 23.26320086655941, 
  						   'obs_days': 57070.03919157, 
  						   'seeing': 1.8900000000000001, #[pixels]
  						   'sky_brightness': 279.6861572265625, #[ADU]
  						   'zero_point': 25.396389422104413}

# Point spread functions to sample
camera_obs_cond[psf].shape = (21, 21, 2312) #Not necessary for light curves

```

A secondary product is a field.dat file for [Surveysim](https://github.com/fforster/surveysim) used as an observation plan for simulating supernovae light curves. One of this file is generated per field an has the following format:

```
MJD FILTER EXPTIME AIRMASS EPOCH
57070.03919157 r 86.0 1.6 1.0
57070.1072727 g 86.0 1.19 2.0
57070.1921733 g 86.0 1.1 3.0
57070.2608982 g 86.0 1.27 4.0
57071.03332127 g 86.0 1.64 5.0
57071.03464703 r 86.0 1.62 6.0
57071.03596559 g 86.0 1.61 7.0
...
```

### Light Curve Sampling

The inly thing you need to have for this part are the obs days in MJD and the limmit of magnitude 

### Image Sequence Simulation




## Packages needed

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

   
