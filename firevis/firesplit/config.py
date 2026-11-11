"""The constants fire tracking uses, copied from firepred/config/dataset.py (fire-split-zones branch).

Only the values fire_tracks.py and fire_zones.py import live here, unchanged,
so the copied modules run as they do upstream. Change them upstream first.
"""

# Input days per window. The TS-SatFire paper's prediction benchmark uses 6.
TS_LENGTH = 6

# ----- Fire split zones ----- #

# One crop can hold several fires: a new ignition far from the main fire, or one
# fire whose front breaks into two. data/fire_tracks.py follows each fire day by
# day from the window archives' labels, and data/fire_zones.py turns that into
# per-window zone files zones/p<id>.npz. One distance sets every threshold:
# gaps smaller than it are noise to bridge, gaps larger than it separate two
# fires. Smaller = more splits; larger = more cleaning. 2.25 km matches FIRED's
# MTBS-tuned 2.3 km and stays above 2x the 0.8 km VIIRS edge-of-scan pixel. [ours]
FIRE_SEPARATION_KM = 2.25
VIIRS_PIXEL_KM = 0.375
# Spread rates that turn the separation distance into days. Slow (0.2 km/day,
# FIRED 2.3 km / 11 days and FEDS 1 km / 5 days agree) sets how long a quiet
# fire is remembered; fast (1 km/day, FEDS forest buffer, Paugam 2026) sets how
# many recent days count as a fire's front when testing for a split.
SLOW_SPREAD_KM_PER_DAY = 0.2
FAST_SPREAD_KM_PER_DAY = 1.0
# Days a new fire must grow on (and a split must hold) before it counts.
MIN_GROWTH_DAYS = 2
# Before the first target day the labels say nothing, so night I4 hotter than
# this marks early fire pixels ("seeds"; night background is about 290 K). They
# count like burned pixels; scattered warm pixels never group into a fire.
SEED_NIGHT_I4_KELVIN = 310.0
NIGHT_I4_CHANNEL = 6
ZONE_FORMAT_VERSION = 1
