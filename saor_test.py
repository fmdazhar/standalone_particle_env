import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1) Synthetic conical pile generator
def generate_conical_pile(num_points=100000, radius=5.0, height=3.0, noise=0.5):
    rs = np.random.rand(num_points)
    thetas = 2 * np.pi * np.random.rand(num_points)
    zs = height * np.random.rand(num_points)
    local_r = (1.0 - zs / height) * radius
    actual_r = rs * local_r
    xs = actual_r * np.cos(thetas)
    ys = actual_r * np.sin(thetas)
    xs += noise * (2 * np.random.rand(num_points) - 1)
    ys += noise * (2 * np.random.rand(num_points) - 1)
    zs += noise * (2 * np.random.rand(num_points) - 1)
    return np.column_stack((xs, ys, zs))

class ReposeAngleCalculatorImproved:
    def __init__(
        self,
        points_3d,
        cylinder_angle_deg=7.5,
        radial_bin_size=0.2,
        height_quantile=0.80,
        drop_fraction=0.10,
        outlier_sigma=2.0,
    ):
        self.p = points_3d
        self.ca = cylinder_angle_deg
        self.rb = radial_bin_size
        self.q = height_quantile
        self.df = drop_fraction
        self.os = outlier_sigma

        self.ang = np.degrees(np.arctan2(self.p[:, 1], self.p[:, 0]))
        self.ang[self.ang < 0] += 360.0
        self.rad = np.sqrt(self.p[:, 0] ** 2 + self.p[:, 1] ** 2)
        self.z = self.p[:, 2]

        self.n_slices = int(round(360.0 / self.ca))
        self.max_bins = int(np.ceil(self.rad.max() / self.rb))

    def _quantile_map(self):
        buckets = [[[] for _ in range(self.max_bins)] for _ in range(self.n_slices)]
        si = (self.ang // self.ca).astype(int)
        ri = (self.rad // self.rb).astype(int)

        mask = (ri < self.max_bins) & (si < self.n_slices)
        for z, s, r in zip(self.z[mask], si[mask], ri[mask]):
            buckets[s][r].append(z)

        qm = np.zeros((self.n_slices, self.max_bins))
        for s in range(self.n_slices):
            for r in range(self.max_bins):
                if buckets[s][r]:
                    qm[s, r] = np.quantile(buckets[s][r], self.q)
        return qm

    def _slice_angle(self, r_centers, h_profile):
        peak = np.argmax(h_profile)
        if h_profile[peak] == 0.0:
            return None
        keep = h_profile >= self.df * h_profile[peak]
        keep &= np.arange(len(h_profile)) >= peak  # downslope only
        idx = np.where(keep)[0]
        if idx.size < 2:
            return None
        m, _ = np.polyfit(r_centers[idx], h_profile[idx], 1)
        return np.degrees(np.arctan(abs(m)))

    def compute(self):
        qm = self._quantile_map()
        edges = np.arange(self.max_bins + 1) * self.rb
        centers = 0.5 * (edges[:-1] + edges[1:])

        slice_angles = [
            a
            for s in range(self.n_slices)
            if (a := self._slice_angle(centers, qm[s])) is not None
        ]
        if not slice_angles:
            return None, None, np.array([])

        A = np.asarray(slice_angles)
        mu, sigma = A.mean(), A.std()
        clean = A[np.abs(A - mu) < self.os * sigma]
        return clean.mean(), clean.std(), clean


# --- Run demo -------------------------------------------------------------
np.random.seed(0)  # reproducibility
pts = generate_conical_pile()
calc = ReposeAngleCalculatorImproved(pts)
angle, spread, per_slice = calc.compute()

print(f"Improved angle of repose: {angle:.2f} ± {spread:.2f}° "
      f"(ground truth 30.96°)")

# 3‑D scatter of point cloud
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Synthetic conical pile")
plt.show()

# Histogram of per‑slice angles
plt.figure(figsize=(6, 4))
plt.hist(per_slice, bins=20)
plt.xlabel("Slice angle [°]")
plt.ylabel("Frequency")
plt.title("Distribution of slice‑wise repose angles")
plt.show()
