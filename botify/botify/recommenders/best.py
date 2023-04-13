from .random import Random
from .toppop import TopPop
from .recommender import Recommender
import random


class Best(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, catalog, history, recommendations):
        self.tracks_redis = tracks_redis
        self.fallback = TopPop(tracks_redis, catalog.top_tracks[:10])
        self.catalog = catalog
        self.history = history
        self.personal_recs = recommendations

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user not in self.history:
            self.history[user] = set()

        previous_track = self.tracks_redis.get(prev_track)

        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        if user not in self.personal_recs:

            previous_track = self.catalog.from_bytes(previous_track)
            recommendations = previous_track.recommendations

            if not recommendations:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

            self.personal_recs[user] = recommendations

        for track in self.personal_recs[user]:
            if track not in self.history[user]:
                self.history[user].add(track)
                return track

        return self.fallback.recommend_next(user, prev_track, prev_track_time)
