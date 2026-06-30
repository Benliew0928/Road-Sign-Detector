# Dataset Licence Acceptance Policy

## Purpose

Every image, video, annotation, and model weight used by
RoadSign Assist must have documented permission for the intended academic use.
An accessible download is not sufficient evidence of permission.

## Accepted Sources

A source may be imported when all of the following are true:

1. The owner or publisher is identified.
2. The source page and immutable identifier, such as a DOI, are recorded.
3. The licence explicitly permits copying, modification, and model training.
4. Attribution and redistribution obligations can be satisfied.
5. The original archive checksum and download date can be recorded.
6. Geographic relevance and label quality are reviewed.

Preferred licences are CC0, CC BY, and compatible open-data licences.

## Review-Required Sources

The source remains `review_required` when:

- The licence is unclear or only visible after account authentication.
- Commercial-use or redistribution restrictions may affect packaging.
- An API grants inference access but not dataset download rights.
- Labels are community-generated without documented quality control.
- Images may contain faces, number plates, or other personal information.

Review-required data cannot enter a frozen training or evaluation release.

## Rejected Sources

Do not import:

- Search-engine or social-media images without explicit reusable licences.
- Data obtained by bypassing authentication, rate limits, or access controls.
- Sources that prohibit derivative works or machine-learning use.
- Personal recordings without the required consent and anonymization.
- Data whose owner, provenance, or class definitions cannot be established.

## Required Provenance Evidence

For every accepted source, preserve:

- Source name, owner, publisher, URL, DOI, and licence URL.
- Download or metadata-retrieval date.
- Archive URL, byte size, and publisher-provided checksum.
- Local SHA-256 checksum after complete download.
- Original class definitions and reviewed imported mapping.
- Accepted/rejected sample counts.
- Attribution text and usage restrictions.

## Data Separation

- The 84 official coursework images are external acceptance data.
- They must never be used for training, augmentation, threshold selection, or
  model selection.
- Duplicate and near-duplicate checks must run before a release is frozen.
- Synthetic data must be identified and must not dominate a supported class.

## Privacy

Faces and number plates must be blurred before data enters an accepted release.
Raw private footage must remain outside Git and be stored only as long as
needed to create the anonymized derivative.
