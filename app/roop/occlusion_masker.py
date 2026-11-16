# SPDX-License-Identifier: MIT
"""Offline semantic + structural residual occlusion, with backward optical flow.

Only CPU OpenCV (Apache-2.0), NumPy (BSD) and existing parser labels are used.
State is bounded and keyed by target track; gaps/cuts must not propagate masks.
This is a conservative heuristic, not a guarantee for every unseen occluder.
"""
from collections import OrderedDict
from threading import RLock

import cv2
import numpy as np


class OcclusionMasker:
    def __init__(self, max_tracks=32):
        self.enabled = True
        self._ordered = True
        self.max_tracks = max_tracks
        self._state = OrderedDict()
        self._lock = RLock()
        self.flow_calls = 0

    def set_ordered(self, ordered):
        self._ordered = bool(ordered)

    def clone_for_block(self):
        return type(self)(self.max_tracks)

    def reset(self):
        with self._lock:
            self._state.clear()

    def compute(self, target, swap, silhouette, *, labels=None,
                preserve_glasses=True, track_id=None, frame_index=None):
        """Return float32 M_occluder, in the aligned target crop's coordinates."""
        h, w = target.shape[:2]
        # Bounded CPU flow/covariance work even for 1024/2048 enhancers. The
        # resulting probability is resampled once into the paste coordinates.
        if max(h, w) > 192:
            ratio = 192. / max(h, w)
            small = (max(1, round(w*ratio)), max(1, round(h*ratio)))
            result = self.compute(cv2.resize(target, small, interpolation=cv2.INTER_AREA),
                                  cv2.resize(swap, small, interpolation=cv2.INTER_AREA),
                                  cv2.resize(np.asarray(silhouette, np.float32), small),
                                  labels=labels, preserve_glasses=preserve_glasses,
                                  track_id=track_id, frame_index=frame_index)
            return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        support = cv2.resize(np.asarray(silhouette, np.float32), (w, h)) > .01
        swap = cv2.resize(swap, (w, h))
        lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        other = cv2.cvtColor(swap, cv2.COLOR_BGR2GRAY)
        semantic = np.zeros((h, w), bool)
        protected = np.zeros_like(semantic)
        skin = support.copy()
        if labels is not None:
            labels = cv2.resize(np.asarray(labels, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            semantic = np.isin(labels, [16, 17, 18] + ([6] if preserve_glasses else []))
            # Eye/brow/nose/mouth labels are authoritative natural features.
            protected = np.isin(labels, [2, 3, 4, 5, 10, 11, 12, 13] + ([] if preserve_glasses else [6]))
            skin &= labels == 1
        residual = np.zeros((h, w), np.float32)
        samples = lab[skin]
        if len(samples) >= 32:
            median = np.median(samples, axis=0)
            # Trim the cluster so an intruding object does not train itself in.
            dist = np.linalg.norm(samples - median, axis=1)
            samples = samples[dist <= np.percentile(dist, 70)]
            cov = np.cov(samples.T) + np.diag([36., 9., 9.])
            delta = lab - np.median(samples, axis=0)
            mahal = np.einsum('...i,ij,...j->...', delta, np.linalg.pinv(cov), delta)
            structural = cv2.absdiff(gray, other).astype(np.float32)
            edges = cv2.Canny(gray, 35, 90)
            edges = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
            candidate = ((mahal > 9) & (structural > 12) & support & ~protected).astype(np.uint8)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Large connected intrusions with a structural boundary only. Small
            # enclosed features (nostrils, beard texture) do not punch holes.
            n, components, stats, _ = cv2.connectedComponentsWithStats(candidate)
            boundary = support & ~cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            for k in range(1, n):
                region = components == k
                if stats[k, cv2.CC_STAT_AREA] < max(16, h*w*.004):
                    continue
                if not (region & edges).any():
                    continue
                # Without semantic protection only edge-connected intrusions
                # are admitted: a dark eye/lip inside skin is not an object.
                if labels is None and not (region & boundary).any():
                    continue
                residual[region] = 1.
        current = np.maximum(semantic.astype(np.float32), residual) * support
        with self._lock:
            old = self._state.get(track_id) if track_id is not None and self._ordered else None
            writable = old is None or frame_index is None or frame_index > old[0]
            if (old is not None and frame_index is not None and frame_index == old[0] + 1
                    and old[1].shape == gray.shape and np.mean(cv2.absdiff(gray, old[1])) < 45):
                # Backward flow maps CURRENT coordinates into the previous map.
                flow = cv2.calcOpticalFlowFarneback(gray, old[1], None, .5, 3, 15, 3, 5, 1.2, 0)
                self.flow_calls += 1
                yy, xx = np.mgrid[:h, :w].astype(np.float32)
                prior = cv2.remap(old[2], xx+flow[..., 0], yy+flow[..., 1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
                # Current evidence stays authoritative; history only softens
                # a one-frame dropout at the advected object boundary.
                current = np.maximum(current, .35 * prior)
            current *= support & ~protected
            if self._ordered and track_id is not None and frame_index is not None and writable:
                self._state[track_id] = (int(frame_index), gray.copy(), current.copy())
                self._state.move_to_end(track_id)
                while len(self._state) > self.max_tracks:
                    self._state.popitem(last=False)
        return np.clip(current, 0, 1).astype(np.float32)
