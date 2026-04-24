"""
Observatory site configurations for ASTROF's three-telescope network.
Each site gets a distinct EarthLocation used by _TelescopeCore for
astropy altitude calculations.
"""
from datetime import datetime

import astropy.units as u
from astropy.coordinates import EarthLocation

SITE_CONFIGS = {
    "mauna_kea": {
        "display_name": "Mauna Kea",
        "location": EarthLocation(lat=19.82 * u.deg, lon=-155.47 * u.deg, height=4207 * u.m),
        "sunset": datetime(2025, 3, 15, 18, 30),
        "sunrise": datetime(2025, 3, 16, 6, 0),
        "sky": "northern",      # can see northern + equatorial targets
    },
    "la_palma": {
        "display_name": "La Palma",
        "location": EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2396 * u.m),
        "sunset": datetime(2025, 3, 15, 19, 15),
        "sunrise": datetime(2025, 3, 16, 6, 45),
        "sky": "northern",
    },
    "siding_spring": {
        "display_name": "Siding Spring",
        "location": EarthLocation(lat=-31.27 * u.deg, lon=149.07 * u.deg, height=1165 * u.m),
        "sunset": datetime(2025, 3, 15, 19, 0),
        "sunrise": datetime(2025, 3, 16, 5, 30),
        "sky": "southern",      # can see southern + equatorial targets
    },
}

TELESCOPE_IDS = list(SITE_CONFIGS.keys())
