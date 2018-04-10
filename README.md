    
# Image Sequence Simulator for HiTS Survey

Code designed to simulate light curves and sequence of images for a certain observation plan and telescope. In this case it is for the HiTS Survey (link), but it will be generalized for other surveys.

## Instructions, inputs and outputs of each step

The code is currently divided in three steps. Get the observation plan, observation conditions and camera parameters, light curve sampling and image sequence generation. In order to use each part, run the following:

### Observation plan, observation conditions and camera parameters

Getting real data from HiTS images. This part is not needed if you build your own camera and obs condition file.

'''
python DataFromHits.py # It will read real supernovae images and get supernovae stamps sequence, observation conditions and camera parameters, generating a pickle file "sn_data.pkl"
sn_data.keys() = ["HiTS01SN", "HiTS02SN", ...]
sn_data["HiTS01SN"].keys() = ['images', 'diff', 'headers', 'psf']
sn_data["HiTS01SN"]["images"].shape = (21, 21, 26) = (x_size, y_size, epochs) # Same format for 'diff' and 'psf'
sn_data["HiTS01SN"]["headers"].keys() = ['obs_days', 'sky_brightness', 'sky_sigma', 'airmass', 'ccd_num', 'exp_time', 'gain', 'seeing', 'pixel_scale', 'read_noise', 'saturation']
'''
The final product of this part is camera_and_obs_cond.pkl, you could build your oun observation plan using a different survey as long as you keep the format of this file. In this case we do the following:

'''
python generate_obs_conditions.py # Generate camera and obs condition file using HiTS data
'''

### Light Curve Sampling

### Image Sequence Simulation




## Packages needed

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

   
