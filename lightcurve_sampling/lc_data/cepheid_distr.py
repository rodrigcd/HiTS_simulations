import numpy as np
import matplotlib.pyplot as plt

data = np.load("cepheid_gps.pkl", encoding="latin1")
stats = data["stats"]

mag_list = []
for stat in stats:
    mag_list.append(stat["g"][0])

print(np.amin(mag_list), np.amax(mag_list))

bins = np.arange(18, 23.5, step=0.1)
ceph_h, _ = np.histogram(mag_list, bins=bins, density=True)

plt.plot(bins[1:], ceph_h)
plt.title("M33 Ceph distribution")
plt.xlabel("magnitude")
plt.show()
