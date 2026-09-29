"""Dataset adapters for image-based position self-guidance.

They preserve the original tensor interface and only attach the raw report
string.  The segmentation model ignores this extra field; the position-guided
wrapper uses it to remove/complete location expressions.
"""

from utils.dataset import MosMed, QaTa
from utils.dataset_semi import (
    MosMedSemiLabeled,
    MosMedSemiUnlabeled,
    QaTaSemiLabeled,
    QaTaSemiUnlabeled,
)


class QaTaPositionLabeled(QaTaSemiLabeled):
    """QaTa labeled adapter that retains the report string for ISPG."""

    def __getitem__(self, index):
        """Attach the raw report without changing the segmentation sample."""
        (inputs, target) = super().__getitem__(index)
        image, text = inputs
        text = dict(text)
        text["raw_text"] = self.caption_list[index]
        return ([image, text], target)


class QaTaPositionUnlabeled(QaTaSemiUnlabeled):
    """QaTa unlabeled adapter that retains the report string for ISPG."""

    def __getitem__(self, index):
        """Attach the raw report to weak/strong unlabeled views."""
        sample = super().__getitem__(index)
        sample["raw_text"] = self.caption_list[index]
        return sample


class QaTaPositionEval(QaTa):
    """QaTa evaluation adapter used by position-removal robustness tests."""

    def __getitem__(self, index):
        """Return the standard sample together with its raw report."""
        (inputs, target) = super().__getitem__(index)
        image, text = inputs
        text = dict(text)
        text["raw_text"] = self.caption_list[index]
        return ([image, text], target)


class MosMedPositionLabeled(MosMedSemiLabeled):
    """MosMed labeled adapter that retains the report string for ISPG."""

    def __getitem__(self, index):
        """Attach the raw report without changing the segmentation sample."""
        (inputs, target) = super().__getitem__(index)
        image, text = inputs
        text = dict(text)
        text["raw_text"] = self.caption_list[index]
        return ([image, text], target)


class MosMedPositionUnlabeled(MosMedSemiUnlabeled):
    """MosMed unlabeled adapter with raw and augmented report strings."""

    def __getitem__(self, index):
        """Expose report strings needed by removal-aware T-PatchMix."""
        sample = super().__getitem__(index)
        sample["raw_text"] = self.caption_list[index]
        # MosMed's original semi-supervised adapter returns tokenized student
        # text only, while T-PatchMix also needs the exact augmented string.
        sample["text_s_str"] = self.tokenizer.decode(
            sample["text_s"]["input_ids"], skip_special_tokens=True
        )
        return sample


class MosMedPositionEval(MosMed):
    """MosMed evaluation adapter used by ISPG robustness tests."""

    def __getitem__(self, index):
        """Return the standard sample together with its raw report."""
        (inputs, target) = super().__getitem__(index)
        image, text = inputs
        text = dict(text)
        text["raw_text"] = self.caption_list[index]
        return ([image, text], target)
