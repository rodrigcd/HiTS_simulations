import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

# Generate fake data
np.random.seed(0)
#y, x = np.mgrid[:128, :128]
x, y= np.mgrid[-21/2.: 21/2., -21/2.: 21/2.] + 0.5

model = models.Gaussian2D(amplitude=100, x_mean=0, y_mean=0,
                          x_stddev=2.0, y_stddev=2.0)
z = np.random.poisson(model(x, y)) + np.random.normal(0., 5, z.shape)
print(z.shape)

# Fit the data using astropy.modeling
p_init = models.Gaussian2D(amplitude=80, x_mean=0.1, y_mean=-0.1,
                           x_stddev=2.0, y_stddev=2.0)
fit_p = fitting.LevMarLSQFitter()

with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
    warnings.simplefilter('ignore')
    p = fit_p(p_init, x, y, z)

print(p)
# Plot the data with the best-fit model
plt.figure(figsize=(8, 2.5))
plt.subplot(1, 3, 1)
plt.imshow(z, origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Data")
plt.subplot(1, 3, 2)
plt.imshow(p(x, y), origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Model")
plt.subplot(1, 3, 3)
plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
plt.colorbar()
plt.title("Residual")
plt.show()