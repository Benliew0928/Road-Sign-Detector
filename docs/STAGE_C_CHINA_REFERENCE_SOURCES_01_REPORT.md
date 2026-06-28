# Stage C China/GB Reference Sources 01 Report

Generated: 2026-06-28

## Purpose

This sprint targets the rare assignment-style signs that are not well covered by
the Malaysian Roboflow sources. Visual review showed these signs match
Mainland China / GB-style road-sign artwork more closely than ordinary Malaysia
road-scene photos.

The goal is to collect traceable official-style reference diagrams for class
anchoring, not to count them as real road-scene training coverage.

## Sources

Primary discovery source:

- `Road signs in China` page, especially the warning, prohibitory, and
  indicative gallery labels.

Downloaded source files:

- Wikimedia Commons per-file road-sign SVG/PNG files listed in
  `data/manifests/stage_c_china_reference_sources_01.csv`.

Local raw/reference files:

- `data/raw/online_sources/stage_c_china_reference_01/wikimedia_commons/`

Review sheets:

- `outputs/review/stage_c_china_reference_sources_01/`

## Result

Downloaded reference candidates:

- 19 reference candidates
- 19 classes with downloaded China/GB-style references
- 18 high-confidence class anchors
- 1 medium-confidence possible style difference
- 0 download failures

High-confidence references:

- `bicycle_crossing`
- `no_left_or_right_turn`
- `no_motor_vehicles`
- `no_straight_or_left`
- `no_straight_or_right`
- `residential_area_ahead`
- `roundabout_mandatory`
- `slow_text`
- `sound_horn`
- `steep_descent`
- `stop_for_checking`
- `straight_ahead`
- `straight_or_right`
- `turn_left`
- `turn_left_or_right`
- `turn_right`
- `uneven_road`
- `width_restriction`

Possible style difference:

- `motor_vehicles_only`: China `Lane for automobiles` reference is a real
  official-style automobiles-only sign, but it is rectangular lane-style
  artwork, so it should stay marked medium-confidence until compared against
  the assignment source image.

## Still Unresolved

- `no_lane_changing`: reliable searches found lane markings / rule references,
  not a clearly matching standalone sign board in the China gallery. Do not map
  this class until the exact source is confirmed.

## Important Limit

These files are official-style reference diagrams. They are useful for:

- Confirming the meaning of rare assignment signs
- Building clean class anchors
- Supporting controlled augmentation or reference-based experiments
- Improving class naming and UI/action logic

They are not independent real road-scene photos. They must not be counted as
final realistic photo coverage in `post_stage_c_realistic_gap_report.*`.

## Main Artifacts

- `scripts/collect_stage_c_china_reference_sources.py`
- `data/manifests/stage_c_china_reference_sources_01.csv`
- `outputs/audit/stage_c_china_reference_sources_01.json`
- `outputs/review/stage_c_china_reference_sources_01/_all_china_reference_candidates.jpg`
