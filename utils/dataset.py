import ast
import json
import os

import torch
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandZoomd,
    Resized,
    ToTensord,
)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            full_data = json.load(f)

        assert self.mode in full_data, f"Mode {self.mode} not found in annotation file."

        self.data = full_data[self.mode]
        self.image_list = [item['image_path'] for item in self.data]
        self.caption_list = [item['caption'] for item in self.data]
        self.pseudo_label_list = [item['pseudo_label'] for item in self.data]

        print(f"{self.mode} set: {len(self.image_list)} samples")

        self.root_path = root_path
        self.image_size = image_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True
        )

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path, 'masks', self.image_list[idx])
        caption = self.caption_list[idx]
        pseudo_label = self.pseudo_label_list[idx]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image, gt, token,mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt==255,1,0)

        text = {
            'input_ids': token.squeeze(0),
            'attention_mask': mask.squeeze(0),
            'pseudo_label': torch.tensor(pseudo_label, dtype=torch.int64),
        }

        return ([image, text], gt)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans


class MosMed(Dataset):
    """MosMedData+ adapter with the same return contract as :class:`QaTa`.

    Images are read from ``frames/`` (or ``Images/`` for the original layout),
    masks from ``masks/``, and reports from a CSV with ``Image`` and ``text``
    columns.  The optional ``pseudo_label`` column stores the six anatomical
    regions used by PACL (Eq. (9)).
    """

    def __init__(self, csv_path=None, root_path=None, tokenizer=None,
                 mode='train', image_size=[224, 224], max_txt_len=24):
        """
        Parameters
        ----------
        csv_path : str
            Path to the MosMedData+ CSV annotations.
        root_path : str
            Dataset root containing ``frames/`` and ``masks/``.
        tokenizer : str or path
            Hugging Face tokenizer identifier or local path.
        mode : str
            ``train``, ``valid``, or ``test``; controls image augmentation.
        image_size : list[int,int]
            Spatial output size, defaulting to ``[224, 224]``.
        max_txt_len : int
            Maximum report-token length, defaulting to 24.
        """
        super().__init__()
        assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"
        self.mode = mode
        self.root_path = root_path
        self.image_size = image_size
        self.max_txt_len = max_txt_len

        # Read CSV or Excel annotations.
        if csv_path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(csv_path)
        else:
            # utf-8-sig accepts spreadsheets exported with a UTF-8 BOM.
            df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Normalize column names and locate the required fields.
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}

        image_col = col_map.get("image")
        text_col = col_map.get("text")
        pseudo_col = col_map.get("pseudo_label")

        if image_col is None or text_col is None:
            raise KeyError(
                "CSV must contain 'Image' and 'text' columns; "
                f"found {list(df.columns)}"
            )

        keep_cols = [image_col, text_col] + ([pseudo_col] if pseudo_col is not None else [])
        df = df[keep_cols]

        df = df.dropna(subset=[image_col, text_col]).copy()
        df[image_col] = df[image_col].astype(str).str.strip()
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df[(df[image_col] != "") & (df[text_col] != "")]

        # Materialize fields as lists for deterministic index-based access.
        self.image_list = df[image_col].tolist()  # e.g., 'bjorke_1.png'
        self.caption_list = df[text_col].tolist()  # e.g., 'Bilateral pulmonary infection, ...

        # The six-dimensional report-derived position label is optional.
        self.pseudo_label_list = None
        if pseudo_col is not None:
            def _parse_plabel(x):
                if x is None:
                    return None
                if isinstance(x, str):
                    x = x.strip()
                    if not x:
                        return None
                    try:
                        v = ast.literal_eval(x)
                    except Exception:
                        return None
                else:
                    v = x
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return [int(i) for i in v]
                return None

            self.pseudo_label_list = [_parse_plabel(x) for x in df[pseudo_col].tolist()]


        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True
        )

        print(f"{self.mode} set (MosMed): {len(self.image_list)} samples")

    def __len__(self):
        """Return the number of image--report pairs."""
        return len(self.image_list)

    def __getitem__(self, idx):
        trans = self.transform(self.image_size)

        # MosMed images and masks share filenames but use separate folders.
        fname = self.image_list[idx]  # e.g., 'bjorke_1.png'

        # Support both the released ``frames/`` and legacy ``Images/`` layouts.
        image = os.path.join(self.root_path, 'frames', fname)
        if not os.path.exists(image):
            alt = os.path.join(self.root_path, 'Images', fname)
            if os.path.exists(alt):
                image = alt
        gt = os.path.join(self.root_path, 'masks', fname)

        # Use the same report encoding contract as QaTa-COV19.
        caption = self.caption_list[idx]
        token_output = self.tokenizer.encode_plus(
            caption, padding='max_length', max_length=24, truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        token, attn = token_output['input_ids'], token_output['attention_mask']

        # Load, resize, normalize, and tensorize with MONAI.
        data = {'image': image, 'gt': gt, 'token': token, 'mask': attn}
        data = trans(data)

        image, gt, token, attn = data['image'], data['gt'], data['token'], data['mask']

        # Normalize grayscale or RGB masks to a single binary channel.
        if gt.ndim == 3 and gt.shape[0] > 1:
            gt = gt.float().mean(dim=0, keepdim=True)
        if gt.max() > 1.5:
            gt = (gt >= 128).to(torch.int64)
        else:
            gt = (gt > 0.5).to(torch.int64)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)

        # Remove the tokenizer's singleton batch dimension.
        text = {'input_ids': token.squeeze(0), 'attention_mask': attn.squeeze(0)}
        if self.pseudo_label_list is not None:
            pseudo_label = self.pseudo_label_list[idx]
            if pseudo_label is not None:
                text['pseudo_label'] = torch.tensor(pseudo_label, dtype=torch.int64)

        return [image, text], gt

    def transform(self, image_size=[224, 224]):
        """Build train or evaluation transforms using the QaTa contract.

        Training adds a light random zoom. Validation and testing use only
        deterministic loading, resizing, image normalization, and conversion
        to tensors. Nearest-neighbor interpolation preserves binary masks.
        """
        if self.mode == "train":
            return Compose([
                LoadImaged(["image", "gt"], reader="PILReader"),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(["image", "gt"], min_zoom=0.95, max_zoom=1.2,
                          mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode="bicubic"),
                Resized(["gt"], spatial_size=image_size, mode="nearest"),
                NormalizeIntensityd(["image"], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])
        else:
            return Compose([
                LoadImaged(["image", "gt"], reader="PILReader"),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode="bicubic"),
                Resized(["gt"], spatial_size=image_size, mode="nearest"),
                NormalizeIntensityd(["image"], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])
