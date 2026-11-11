"""Fire splitting from fire-research-and-innovation, branch fire-split-zones (commit dc90aad).

    fire_records.py   SplitSettings, FireEvent, ZoneFile        (firepred/data/fire_records.py)
    fire_tracks.py    track every fire of an archive day by day  (firepred/data/fire_tracks.py)
    fire_zones.py     zones and target owners per window         (firepred/data/fire_zones.py)
    config.py         the constants they use                     (firepred/config/dataset.py)

The three modules are copies of the upstream files with only their
`firepred.*` imports made relative, so a notebook needs neither firepred nor
torch to run the same split. To update: copy the files again from that branch,
redo the imports, and bump the commit above.

firevis/split.py reads an archive and runs these; firevis/split_viewer.py shows the result.
"""

SOURCE_REPO = "GT-Fire-Research-and-Innovation/fire-research-and-innovation"
SOURCE_BRANCH = "fire-split-zones"
SOURCE_COMMIT = "dc90aad"
