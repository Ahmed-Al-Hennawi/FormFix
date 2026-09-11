"""
Maps validation RejectionCodes to the smaller set of failure screens the UI
has, so a recording problem never looks like a technique fault.
"""

from __future__ import annotations

from analysis.models import FailureCode, RejectionCode, ValidationResult

FAILURE_CODES: dict[RejectionCode, FailureCode] = {
    RejectionCode.VIDEO_READ_ERROR: FailureCode.INVALID_VIDEO,
    RejectionCode.VIDEO_TOO_SHORT: FailureCode.INVALID_VIDEO,
    RejectionCode.VIDEO_TOO_LONG: FailureCode.INVALID_VIDEO,
    RejectionCode.LOW_RESOLUTION: FailureCode.INVALID_VIDEO,
    RejectionCode.NO_POSE_DETECTED: FailureCode.NO_POSE,
    RejectionCode.INSUFFICIENT_POSE_COVERAGE: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.IMPORTANT_LANDMARKS_MISSING: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.INSUFFICIENT_VALID_FRAMES: FailureCode.INSUFFICIENT_VISIBILITY,
    RejectionCode.BODY_OUT_OF_FRAME: FailureCode.BODY_OUT_OF_FRAME,
    RejectionCode.MULTIPLE_PEOPLE: FailureCode.MULTIPLE_PEOPLE,
    RejectionCode.UNSUPPORTED_CAMERA_ANGLE: FailureCode.UNSUITABLE_CAMERA_VIEW,
    RejectionCode.NO_SQUAT_MOVEMENT: FailureCode.NO_COMPLETE_SQUAT,
    RejectionCode.NO_COMPLETE_REPETITION: FailureCode.NO_COMPLETE_REPETITION,
}


def failure_code_for(checks: ValidationResult) -> FailureCode:
    """First known rejection code wins, unknown ones fall back."""
    for code in checks.reason_codes:
        mapped = FAILURE_CODES.get(code)
        if mapped is not None:
            return mapped
    return FailureCode.INVALID_VIDEO
