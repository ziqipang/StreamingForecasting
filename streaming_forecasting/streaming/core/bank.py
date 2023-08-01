from .track import Track


class TrackBank:
    def __init__(self) -> None:
        self.frame_idx = 0
        self.tracks = dict() # {id: track}

    def load_new_pos(self, positions, ids, classes):
        track_keys = list(self.tracks.keys())
        for idx, (pos, id, cls) in enumerate(zip(positions, ids, classes)):
            if id in track_keys:
                self.tracks[id].load_new_pos(pos, self.frame_idx)
            else:
                self.tracks[id] = Track(id, cls, pos, self.frame_idx)
        return
    
    def get_tracks(self):
        return self.tracks
    
    def get_active_tracks(self):
        result = dict()
        track_keys = list(self.tracks.keys())
        for id in track_keys:
            trk = self.tracks[id]
            if trk.frames[-1] <= self.frame_idx:
                continue
            else:
                result[id] = trk
        return result