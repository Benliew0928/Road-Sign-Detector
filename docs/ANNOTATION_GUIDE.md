# Road-Sign Annotation Guide

## Geometry

- Draw one polygon around the visible sign face.
- Exclude the sign pole, mounting bracket, sky, and background.
- Keep the associated bounding box tight to the polygon.
- If more than half of the sign face is hidden, mark `severe_occlusion`.
- Do not infer invisible geometry behind an obstruction.

## Semantic Class

- Use only identifiers from `configs/catalogue/malaysia_signs.v1.json`.
- Use `unknown_sign` in review notes when the visible meaning is uncertain.
- Never infer a class from the filename or source folder.
- Preserve coursework IDs separately from semantic classes.

## Text

- Transcribe exactly what is visibly printed.
- Preserve Chinese characters and Malay/English spelling.
- Use uppercase only when the sign itself is uppercase.
- Do not translate inside the transcript field.
- Record numbers and units exactly as displayed.

## Review

1. The annotator submits the task.
2. A different reviewer validates geometry, class, and transcript.
3. Safety-critical signs require a second reviewer.
4. Rejected annotations return to the original annotator with a reason.

