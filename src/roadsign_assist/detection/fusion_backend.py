from __future__ import annotations

from collections.abc import Sequence

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.detection.base import SignDetector
from roadsign_assist.inference.models import DetectionModel


class FusionSignDetector:
    """Run fixed detector members and merge their generic-sign boxes with NMS."""

    def __init__(
        self,
        members: Sequence[SignDetector],
        *,
        nms_iou_threshold: float = 0.50,
        profile_name: str = "fusion",
    ) -> None:
        if len(members) < 2:
            raise ValueError("Fusion requires at least two detector members")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("Fusion NMS IoU threshold must be between 0 and 1")
        self.members = tuple(members)
        self.nms_iou_threshold = nms_iou_threshold
        self.profile_name = profile_name

    @property
    def name(self) -> str:
        return "fusion:" + "+".join(
            str(getattr(member, "profile_name", member.name)) for member in self.members
        )

    @property
    def available(self) -> bool:
        return all(member.available for member in self.members)

    @property
    def loaded(self) -> bool:
        return all(bool(getattr(member, "loaded", True)) for member in self.members)

    @property
    def active_device(self) -> str | None:
        devices = {
            str(value)
            for member in self.members
            if (value := getattr(member, "active_device", None)) is not None
        }
        return "+".join(sorted(devices)) if devices else None

    @property
    def task(self) -> str:
        return "detect"

    @property
    def assignment_only(self) -> bool:
        return any(bool(getattr(member, "assignment_only", False)) for member in self.members)

    @property
    def mask_capable(self) -> bool:
        return False

    def warmup(self) -> bool:
        results = [member.warmup() for member in self.members]
        return all(results)

    def detect(self, image: UInt8Image) -> list[DetectionModel]:
        proposed: list[tuple[int, int, DetectionModel]] = []
        for member_index, member in enumerate(self.members):
            for detection_index, detection in enumerate(member.detect(image)):
                proposed.append((member_index, detection_index, detection))

        # Generic road-sign detection is class agnostic. Stable tie-breaking keeps
        # the result reproducible and gives the first member precedence on ties.
        proposed.sort(key=lambda item: (-item[2].confidence, item[0], item[1]))
        retained: list[tuple[int, DetectionModel]] = []
        for member_index, _, candidate in proposed:
            if any(
                candidate.bbox.iou(existing.bbox) > self.nms_iou_threshold
                for _, existing in retained
            ):
                continue
            retained.append((member_index, candidate))

        return [
            detection.model_copy(
                update={
                    "detection_id": f"fusion-{index}",
                    "detector": (
                        f"{getattr(self.members[member_index], 'profile_name', self.members[member_index].name)}"
                        "|fusion"
                    ),
                }
            )
            for index, (member_index, detection) in enumerate(retained)
        ]
