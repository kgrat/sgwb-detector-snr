"""Next-generation detector geometry from Gupta et al. (arXiv:2307.10421), Table II.

Conventions (Table II caption):
- latitude positive north, longitude positive east of Greenwich [deg];
- orientation = azimuth of the x-arm, measured counter-clockwise from local
  east ("north of east"), which is also bilby's xarm_azimuth convention;
- L-shaped detectors: x, y and the local outward vertical form a right-handed
  set, so the y-arm is 90 deg counter-clockwise from the x-arm;
- ET: the other two arms lie westward of the x-arm, so ET1's y-arm is 60 deg
  counter-clockwise from its x-arm (x-arm due north, y-arm north-west).
Table II gives no elevation; we use 0 m for every vertex.

The alternative "2L" ET design (two L-shaped detectors, Sardinia and
Meuse-Rhine) follows Branchesi et al. (arXiv:2303.15923), Sec. 2 and App. B.

bilby is used only as a container for vertex and arm unit vectors. Its
built-in CE/ET files are not used (wrong sites), and neither is
bilby.gw.detector.TriangularInterferometer: it places the ET vertices
correctly but rotates ET2's and ET3's arms by -120 deg instead of +120 deg,
so ET2 gets the arm pair ET3 needs and vice versa.
"""

import numpy as np
from astropy.constants import R_earth
from bilby.gw.detector import Interferometer

# Table II of arXiv:2307.10421. Angles in degrees.
_TABLE_II = {
    "CE-A": {"latitude_deg": 46.0, "longitude_deg": -125.0, "x_arm_azimuth_deg": 260.0},
    "CE-B": {"latitude_deg": 29.0, "longitude_deg": -94.0, "x_arm_azimuth_deg": 200.0},
    "ET": {"latitude_deg": 40.0 + 31.0 / 60.0, "longitude_deg": 9.0 + 25.0 / 60.0, "x_arm_azimuth_deg": 90.0},
}
ELEVATION_M = 0.0  # not given in Table II

# Gupta et al. Sec. II: CE-A hosts a 40 km or a 20 km detector, CE-B only 20 km.
_CE_CONFIGURATIONS = {("CE-A", 40.0), ("CE-A", 20.0), ("CE-B", 20.0)}

# ET-D: three interferometers with 10 km arms (Hild et al. 2011, arXiv:1012.0908).
ET_ARM_LENGTH_KM = 10.0
ET_OPENING_ANGLE_DEG = 60.0


def cosmic_explorer(site, arm_length_km, minimum_frequency_hz, maximum_frequency_hz):
    """L-shaped Cosmic Explorer detector at a Table II site ("CE-A" or "CE-B").

    The frequency band is stored by bilby but does not affect the geometry.
    """
    if (site, arm_length_km) not in _CE_CONFIGURATIONS:
        raise ValueError(
            f"No {arm_length_km} km CE at '{site}' in Gupta et al.; "
            f"allowed (site, km): {sorted(_CE_CONFIGURATIONS)}"
        )
    geometry = _TABLE_II[site]
    return Interferometer(
        name=f"CE{arm_length_km:.0f}{site[-1]}",
        power_spectral_density=None,
        minimum_frequency=minimum_frequency_hz,
        maximum_frequency=maximum_frequency_hz,
        length=arm_length_km,
        latitude=geometry["latitude_deg"],
        longitude=geometry["longitude_deg"],
        elevation=ELEVATION_M,
        xarm_azimuth=geometry["x_arm_azimuth_deg"],
        yarm_azimuth=(geometry["x_arm_azimuth_deg"] + 90.0) % 360.0,
    )


def einstein_telescope(minimum_frequency_hz, maximum_frequency_hz):
    """The three 60-degree interferometers ET1, ET2, ET3 of the ET triangle.

    ET1 sits at the Table II vertex. Rotating ET1 by 120 deg about the
    triangle's centre carries vertex k to vertex k+1 and ET1's two arms onto
    the two edges meeting there, so ET(k+1) has x-arm azimuth
    (Table II orientation + 120 k) and its y-arm 60 deg further
    counter-clockwise. Each next vertex is the far end of the current x-arm.
    """
    geometry = _TABLE_II["ET"]
    latitude_deg, longitude_deg = geometry["latitude_deg"], geometry["longitude_deg"]
    interferometers = []
    for k in range(3):
        x_arm_azimuth_deg = (geometry["x_arm_azimuth_deg"] + 120.0 * k) % 360.0
        interferometers.append(
            Interferometer(
                name=f"ET{k + 1}",
                power_spectral_density=None,
                minimum_frequency=minimum_frequency_hz,
                maximum_frequency=maximum_frequency_hz,
                length=ET_ARM_LENGTH_KM,
                latitude=latitude_deg,
                longitude=longitude_deg,
                elevation=ELEVATION_M,
                xarm_azimuth=x_arm_azimuth_deg,
                yarm_azimuth=(x_arm_azimuth_deg + ET_OPENING_ANGLE_DEG) % 360.0,
            )
        )
        latitude_deg, longitude_deg = _great_circle_step(
            latitude_deg, longitude_deg, x_arm_azimuth_deg, ET_ARM_LENGTH_KM * 1e3
        )
    return interferometers


# Branchesi et al. Sec. 2, footnote 3. Angles in degrees.
_ET_2L_SITES = {
    "Sardinia": {"latitude_deg": 40.0 + 31.0 / 60.0, "longitude_deg": 9.0 + 25.0 / 60.0},
    "Meuse-Rhine": {
        "latitude_deg": 50.0 + 43.0 / 60.0 + 23.0 / 3600.0,
        "longitude_deg": 5.0 + 55.0 / 60.0 + 14.0 / 3600.0,
    },
}


def einstein_telescope_2l(relative_angle_deg, arm_length_km, minimum_frequency_hz, maximum_frequency_hz):
    """The two L-shaped detectors of the ET 2L design (Branchesi et al. App. B).

    The Sardinia arms point locally east (x) and north (y). The Meuse-Rhine
    arms are rotated counter-clockwise by relative_angle_deg (their alpha) from
    local east/north. Branchesi et al. use alpha = 0 ("parallel") and 45 deg;
    because of the Earth's curvature the truly parallel and maximally
    misaligned orientations are alpha = 2.51 and 47.51 deg.
    """
    detectors = []
    for site, x_arm_azimuth_deg in (("Sardinia", 0.0), ("Meuse-Rhine", relative_angle_deg)):
        geometry = _ET_2L_SITES[site]
        detectors.append(
            Interferometer(
                name=f"ET2L{site[0]}",
                power_spectral_density=None,
                minimum_frequency=minimum_frequency_hz,
                maximum_frequency=maximum_frequency_hz,
                length=arm_length_km,
                latitude=geometry["latitude_deg"],
                longitude=geometry["longitude_deg"],
                elevation=ELEVATION_M,
                xarm_azimuth=x_arm_azimuth_deg % 360.0,
                yarm_azimuth=(x_arm_azimuth_deg + 90.0) % 360.0,
            )
        )
    return detectors


def great_circle_azimuth_deg(latitude_1_deg, longitude_1_deg, latitude_2_deg, longitude_2_deg):
    """Azimuth (counter-clockwise from local east) at point 1 of the great circle to point 2.

    Standard initial-bearing formula; bearing (clockwise from north) = 90 deg - azimuth.
    """
    phi_1, phi_2 = np.radians(latitude_1_deg), np.radians(latitude_2_deg)
    delta_lambda = np.radians(longitude_2_deg - longitude_1_deg)
    bearing_rad = np.arctan2(
        np.sin(delta_lambda) * np.cos(phi_2),
        np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2) * np.cos(delta_lambda),
    )
    return 90.0 - np.degrees(bearing_rad)


def _great_circle_step(latitude_deg, longitude_deg, azimuth_deg, distance_m):
    """Point reached by walking distance_m along the great circle leaving
    (latitude_deg, longitude_deg) at azimuth_deg (counter-clockwise from east).

    Standard spherical destination-point formula, with bearing measured
    clockwise from north = 90 deg - azimuth. Treating the Earth as a sphere of
    radius R_earth moves a 10 km step's end point by tens of metres relative
    to WGS-84; the resulting ORF phase error, 2 pi f dx / c, stays below
    1e-3 rad at 512 Hz.
    """
    bearing_rad = np.radians(90.0 - azimuth_deg)
    angular_distance_rad = distance_m / R_earth.to_value("m")
    phi_1, lambda_1 = np.radians(latitude_deg), np.radians(longitude_deg)
    phi_2 = np.arcsin(
        np.sin(phi_1) * np.cos(angular_distance_rad)
        + np.cos(phi_1) * np.sin(angular_distance_rad) * np.cos(bearing_rad)
    )
    lambda_2 = lambda_1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance_rad) * np.cos(phi_1),
        np.cos(angular_distance_rad) - np.sin(phi_1) * np.sin(phi_2),
    )
    return np.degrees(phi_2), np.degrees(lambda_2)
