# Malaysian Road-Sign Data Collection Protocol

## Safety Boundary

The camera operator must not drive, cycle, or stand in a traffic lane while
collecting data. Record from a passenger position, a legally parked vehicle,
or a safe pedestrian area. The system is an advisory research prototype and
must not control a vehicle.

## Coverage Plan

Collect separate sessions for:

- Urban roads, residential streets, school zones, and construction zones.
- Federal/state roads, rural roads, and expressway approaches.
- Daylight, dusk, night, rain, glare, shadow, and backlighting.
- Near, medium, small/distant, partially occluded, rotated, and damaged signs.
- Malay, English, Chinese, mixed-language, and numeric signs.
- No-sign scenes containing sign-like objects, advertising, vehicles, and lights.

At least 20% of accepted road-scene frames should contain no target sign.

## Session Metadata

Record:

- `source_id`, `session_id`, `route_id`, date, and broad area.
- Camera model, resolution, orientation, and frame rate.
- Lighting/weather category and road type.
- Collector and reviewer identifiers.
- Whether faces or number plates are present.

Do not store precise private-home locations in the public manifest.

## Capture Guidance

- Prefer 1080p or higher when safe and available.
- Avoid digital zoom; preserve the road context for detection training.
- Keep short continuous sequences so tracking can be evaluated.
- Do not repeatedly sample the same physical sign into different splits.
- Include deliberate difficult cases without endangering the collector.

## Processing

1. Copy raw media into a source/session directory without modifying it.
2. Calculate SHA-256 checksums.
3. Extract candidate frames using a documented interval and blur filter.
4. Blur faces and number plates.
5. Assign physical-sign, route, and session group identifiers.
6. Remove exact and perceptual duplicates.
7. Register accepted/rejected counts and reasons.
8. Import accepted frames into CVAT.

## Privacy and Retention

Raw private footage stays outside Git. Delete it after the anonymized dataset
release is verified unless a documented consent and retention reason exists.

