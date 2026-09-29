"""Labeled and unlabeled dataset views for Semi-MedRef training."""

from typing import Optional

from monai.transforms import Compose

from utils.dataset import MosMed, QaTa
from utils.pos_aug import POS_PHRASE_RE, pos_token_dropout


class QaTaSemiLabeled(QaTa):
    """QaTa labeled split with an externally supplied training transform."""

    def __init__(self, *args, labeled_tf: Optional[Compose] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._labeled_tf = labeled_tf

    def transform(self, image_size=(224, 224)):
        if self._labeled_tf is not None:
            return self._labeled_tf
        return super().transform(image_size)


class QaTaSemiUnlabeled(QaTa):
    """Return paired weak/strong image and report views for unlabeled data."""

    def __init__(
        self,
        *args,
        weak_tf: Compose,
        strong_aug,
        use_pos_aug: bool = False,
        pos_aug_mode: str = "mix",
        pos_aug_p: float = 0.4,
        use_pos_aug_geosync: bool = False,
        **kwargs,
    ):
        if "mode" in kwargs:
            assert kwargs["mode"] == "train", "Unlabeled split should be train."
        super().__init__(*args, **kwargs)
        self._weak_tf = weak_tf
        self._strong_aug = strong_aug
        self.use_pos_aug = use_pos_aug
        self.pos_aug_mode = pos_aug_mode
        self.pos_aug_p = float(pos_aug_p)
        self.use_pos_aug_geosync = use_pos_aug_geosync

    def transform(self, image_size=(224, 224)):
        return self._weak_tf

    def __getitem__(self, idx):
        (image_w, text_tok), gt_w = super().__getitem__(idx)
        image_s = self._strong_aug(image_w)

        raw_text = self.caption_list[idx]
        text_w = text_tok
        text_s_str = raw_text
        has_pos = POS_PHRASE_RE.search(raw_text) is not None

        if self.use_pos_aug:
            text_s_str = pos_token_dropout(
                text_s_str,
                p_mask=self.pos_aug_p,
                mode=self.pos_aug_mode,
            )
        pos_dropout_applied = (
            "[UNK_POS]" in text_s_str or "a region of the lung" in text_s_str
        )

        tokenized = self.tokenizer.encode_plus(
            text_s_str,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_s = {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "pseudo_label": text_tok["pseudo_label"],
        }

        # Geometric report synchronization is retained in the interface for
        # compatibility but disabled in the released main-table recipes.
        return {
            "img_w": image_w,
            "img_s": image_s,
            "text": text_tok,
            "text_w": text_w,
            "text_s": text_s,
            "text_s_str": text_s_str,
            "gt_w": gt_w,
            "geom": {"hflip": False, "vflip": False},
            "flags": {
                "has_pos": has_pos,
                "geosync": False,
                "pos_dropout": pos_dropout_applied,
            },
        }


class MosMedSemiLabeled(MosMed):
    """MosMed labeled split with an externally supplied training transform."""

    def __init__(self, *args, labeled_tf: Optional[Compose] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._labeled_tf = labeled_tf

    def transform(self, image_size=(224, 224)):
        if self._labeled_tf is not None:
            return self._labeled_tf
        return super().transform(image_size)


class MosMedSemiUnlabeled(MosMed):
    """MosMed counterpart of :class:`QaTaSemiUnlabeled`."""

    def __init__(
        self,
        *args,
        weak_tf: Compose,
        strong_aug,
        use_pos_aug: bool = False,
        pos_aug_mode: str = "mix",
        pos_aug_p: float = 0.4,
        **kwargs,
    ):
        if "mode" in kwargs:
            assert kwargs["mode"] == "train", "Unlabeled split should be train."
        super().__init__(*args, **kwargs)
        self._weak_tf = weak_tf
        self._strong_aug = strong_aug
        self.use_pos_aug = use_pos_aug
        self.pos_aug_mode = pos_aug_mode
        self.pos_aug_p = float(pos_aug_p)

    def transform(self, image_size=(224, 224)):
        return self._weak_tf

    def __getitem__(self, idx):
        (image_w, text_tok), gt_w = super().__getitem__(idx)
        image_s = self._strong_aug(image_w)

        text_w = text_tok
        text_s = text_tok
        text_s_str = self.caption_list[idx]
        if self.use_pos_aug:
            text_s_str = pos_token_dropout(
                text_s_str,
                p_mask=self.pos_aug_p,
                mode=self.pos_aug_mode,
            )
            tokenized = self.tokenizer.encode_plus(
                text_s_str,
                padding="max_length",
                max_length=24,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_s = {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
            }
            if "pseudo_label" in text_tok:
                text_s["pseudo_label"] = text_tok["pseudo_label"]

        return {
            "img_w": image_w,
            "img_s": image_s,
            "text": text_tok,
            "text_w": text_w,
            "text_s": text_s,
            "text_s_str": text_s_str,
            "gt_w": gt_w,
        }
