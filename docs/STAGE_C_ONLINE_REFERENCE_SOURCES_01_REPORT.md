# Stage C Online Reference Sources 01 Report

Generated: 2026-06-28

## Purpose

This sprint searches reliable online sources for the remaining Stage C gap
classes without using local phone/camera collection.

The goal is to collect traceable Malaysian road-sign references from online
sources, not to pretend that reference diagrams are the same as real road-scene
photos.

## Source Families

Primary sources used:

- Wikimedia Commons Malaysian road-sign SVG files
- Wikimedia Commons Malaysian warning-sign diagrams
- HuggingFace `wanadzhar913/wikipedia-malaysian-road-sign-images`, which points
  to Wikimedia-hosted Malaysian road-sign image URLs

Source URLs, file titles, licence metadata, and local paths are recorded in:

- `data/manifests/stage_c_online_reference_sources_01.csv`

Raw/reference files are stored under:

- `data/raw/online_sources/stage_c_reference_01/wikimedia_commons/`

Review sheets are stored under:

- `outputs/review/stage_c_online_reference_sources_01/`

## Result

Downloaded online reference candidates:

- 15 downloaded reference candidates
- 11 classes with high-confidence exact or near-exact Malaysian references
- 1 downloaded possible mismatch candidate
- 1 failed redundant download due to Wikimedia 429 rate limiting

High-confidence references found:

- `no_heavy_vehicle`
- `pass_either_side`
- `school_zone`
- `side_road_right`
- `steep_descent`
- `straight_ahead`
- `straight_or_right`
- `turn_left`
- `turn_right`
- `uneven_road`
- `width_restriction`

Possible mismatch, not counted as solved:

- `motor_vehicles_only`: `Malaysia road sign RM3a.svg` shows multiple permitted
  vehicle types and may not match the assignment `motor_vehicles_only` meaning.

## Important Limit

These are online reference candidates, mostly diagrams/SVGs. They are useful
for:

- Confirming exact Malaysian sign appearance
- Building class reference sheets
- Supporting controlled classifier experiments
- Helping future augmentation that is clearly marked as reference-derived

They are not the same as 50 independent real road-scene photos. They do not
close the realistic-photo coverage gap by themselves.

## Still Unresolved From Reliable Online Sources

These classes still need a better source, most likely direct JKR artwork
extraction or another public dataset with exact labels:

- `bicycle_crossing`
- `motor_vehicles_only`
- `no_lane_changing`
- `no_left_or_right_turn`
- `no_motor_vehicles`
- `no_straight_or_left`
- `no_straight_or_right`
- `roundabout_mandatory`
- `slow_text`
- `sound_horn`
- `stop_for_checking`
- `turn_left_or_right`

## Next Reliable Step

The best next online-only route is:

1. Use the downloaded exact references as verified class anchors.
2. Try to acquire or extract the official JKR ATJ 2A/85 sign artwork for the
   unresolved assignment signs.
3. Search additional public datasets only if they expose exact labels and
   licence/provenance.
4. Keep all reference-derived training experiments separated from real-photo
   evaluation.

Audit report:

- `outputs/audit/stage_c_online_reference_sources_01.json`

Combined visual sheet:

- `outputs/review/stage_c_online_reference_sources_01/_all_reference_candidates.jpg`
