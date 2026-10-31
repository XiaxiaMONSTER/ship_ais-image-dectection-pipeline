class Tracker:
    def __init__(self):
        self.tracks = {}   # mmsi -> list of (lat, lon, t)

    def update(self, mmsi, lat, lon, ts):
        if mmsi not in self.tracks:
            self.tracks[mmsi] = []
        self.tracks[mmsi].append((lat, lon, ts))
