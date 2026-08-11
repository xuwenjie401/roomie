# Object evidence pipeline

The online object path treats physical identity, detector semantics, and
continued presence as separate evidence streams:

```text
OWL per-label candidates + BoxerNet OBB/DINO ROI
  -> same-frame physical observation clustering
  -> factor matrix + Hungarian assignment
  -> semantic posterior and bounded presence log-odds
  -> tentative / active / archived
```

`instance.association_mode` controls rollout:

- `legacy`: the previous semantic-gated greedy associator.
- `shadow`: computes physical clusters and Hungarian assignments for logs, but
  leaves legacy identity decisions authoritative.
- `evidence`: makes the new associator and presence lifecycle authoritative.

The checked-in robot configurations select `evidence`; the C++ configuration
default remains `legacy` for embedders that do not load a Roomie YAML file.

## Identity and semantics

OWL remains per-label NMS. Highly overlapping cross-label outputs are combined
after BoxerNet into one physical observation with a soft label distribution,
so `handbag` and `backpack` evidence from one frame cannot create two tracks or
count as two confirmation frames. Global assignment is deterministic and does
not use semantic equality as a gate.

Existing duplicate objects require merge support in two distinct successful
frames. A merge retires the duplicate through a durable alias; it does not
tombstone either historical identity. Human-annotated identity wins canonical
selection, otherwise the older object id wins.

The response wire version is `RIRS3`. Each 3D detection may carry a normalized
`boxernet_dinov3_roi_mean_v1` descriptor from the already-computed DINO feature
map. Appearance similarity is logged and persisted on tracks for shadow
evaluation, but has zero decision weight.

## Presence

Positive evidence comes from an assigned physical observation. A tentative or
archived track needs two successful frames from distinct camera viewpoints
within 20 eligible inference frames before becoming active. The positive
window advances only when InstanceMap processes a successful, admitted
inference response, so frames rejected during robot rotation do not age the
evidence. By default, viewpoints are distinct when their camera origins differ
by at least `clamp(0.15 * object_diagonal, 0.08 m, 0.20 m)` or their
object-relative viewing directions differ by at least 6 degrees.

Two observations from the same viewpoint may also confirm an object when both
are within 2.5 m, have raw confidence at least 0.85 and bbox quality at least
0.80. Lower-quality same-viewpoint repeats still update existence log-odds
and `last_seen`, but do not advance confirmation. Pure camera rotation without
optical-center motion does not create parallax and is therefore not an
independent viewpoint.

For an unmatched active object, the admitted frame's registered raw depth is
tested against projected ray/OBB entry and exit distances:

- out of view, image edge, robot mask, invalid depth, and foreground
  occlusion are unknown and add zero evidence;
- depth inside the OBB is occupied evidence;
- sufficiently sampled depth behind the OBB is free-space evidence.

Archival requires at least three reliable free-space frames within two seconds
and log-odds below the archive threshold. Archived objects disappear from
`/roomie/objects`, remain in `/roomie/instances` and historical queries, and
reuse their original id after two-frame re-identification. Mapping-only runs
without successful BoxerNet responses preserve lifecycle state.

GeometryWorker validation belongs exclusively to `GeometryComponent`.
Association upserts can advance an OBB revision but cannot overwrite an
accepted geometry CAS result with stale fields from `InstanceTrack`.

## Diagnostics

Run logs use these categories:

- `association_shadow`: aggregate shadow decisions;
- `association_decision`: factor values and selected assignment;
- `presence_evidence`: zero/free-space evidence and lifecycle state;
- `instance_map`: raw, physical, clustered, track, merge, and object counts.

All physical clustering, assignment, and presence thresholds are ROS
parameters under `instance.*`; production defaults are recorded in the four
pipeline YAML files.
