# infer.ipynb / infer.py
# Kaggle 提出用（sample_submission.csv を使わない）
# - test.csv から submission（sample_id, target）を作る
# - 重みは Kaggle Dataset にアップロードして /kaggle/input/<weights-dataset>/ から読む
#
# 追加仕様:
# - Lightning ckpt に ema_shadow が入っている場合、推論で EMA 重みを適用できる

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm


# =====================
# 競技仕様（あなたのロジック）
# =====================
TARGET5_ORDER = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]


# =====================
# Dataset / Transform
# =====================
class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_dir: Path, transform: Optional[transforms.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        img_path = self.input_dir / str(row["image_path"])
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def create_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# =====================
# test.csv 整形
# =====================
def make_test_tables(test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    期待する列（どれかが必要）:
    - image_path
    - sample_id もしくは (sample_id_prefix & sample_id_suffix) もしくは target_name
    """
    test_long = pd.read_csv(test_csv)

    if "image_path" not in test_long.columns:
        raise ValueError("test.csv に image_path 列が見つかりません。")

    # sample_id_suffix を作る
    if "sample_id_suffix" in test_long.columns:
        pass
    elif "target_name" in test_long.columns:
        test_long["sample_id_suffix"] = test_long["target_name"].astype(str)
    elif "sample_id" in test_long.columns:
        sp = test_long["sample_id"].astype(str).str.split("__", n=1, expand=True)
        if sp.shape[1] != 2:
            raise ValueError("sample_id を '__' で分割できません（prefix__suffix 形式ではないようです）。")
        test_long["sample_id_prefix"] = sp[0]
        test_long["sample_id_suffix"] = sp[1]
    else:
        raise ValueError("test.csv に sample_id_suffix / target_name / sample_id のどれも見つかりません。")

    # sample_id を作る
    if "sample_id" not in test_long.columns:
        if "sample_id_prefix" in test_long.columns and "sample_id_suffix" in test_long.columns:
            test_long["sample_id"] = (
                test_long["sample_id_prefix"].astype(str) + "__" + test_long["sample_id_suffix"].astype(str)
            )
        else:
            raise ValueError("sample_id を作れません（sample_id_prefix がありません）。")

    # prefix を作る（submission 組み立てのため）
    if "sample_id_prefix" not in test_long.columns:
        sp = test_long["sample_id"].astype(str).str.split("__", n=1, expand=True)
        if sp.shape[1] != 2:
            raise ValueError("sample_id から sample_id_prefix を作れません（prefix__suffix 形式ではないようです）。")
        test_long["sample_id_prefix"] = sp[0]

    # 画像1枚=1行（推論用）
    test_img = (
        test_long[["image_path", "sample_id_prefix"]]
        .drop_duplicates(subset=["image_path"])
        .reset_index(drop=True)
    )
    return test_img, test_long


# =====================
# モデル作成
# =====================
def create_model(model_name: str, num_classes: int = 3) -> nn.Module:
    # 学習側と num_classes が一致している必要あり
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=3)
    return model


# =====================
# 重み / EMA の読み込み補助
# =====================
def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _apply_ema_shadow_to_model(model: nn.Module, ema_shadow: Dict[str, torch.Tensor]) -> None:
    """
    ema_shadow を model に適用する。
    
    重要: ema_shadow は state_dict() と同じ形式で保存されており、
    学習可能なパラメータだけでなく、バッファ（running_mean, running_var など）も含む。
    そのため named_parameters() ではなく load_state_dict() を使う必要がある。
    """
    ema_shadow = _strip_prefix(ema_shadow, prefixes=["model.", "net.", "module."])

    # デバイスを合わせる
    device = next(model.parameters()).device
    ema_shadow_on_device = {
        name: val.to(device) for name, val in ema_shadow.items()
    }

    # load_state_dict を使って適用（バッファも含めて全て適用される）
    missing, unexpected = model.load_state_dict(ema_shadow_on_device, strict=False)
    if unexpected:
        print(f"[warn][ema] unexpected keys: {unexpected[:5]} ... (total {len(unexpected)})")
    if missing:
        print(f"[warn][ema] missing keys: {missing[:5]} ... (total {len(missing)})")


def load_weights_flexible(
    model: nn.Module,
    weight_path: Path,
    use_ema_infer: bool = True,
) -> nn.Module:
    """
    - .pth: state_dict 直 を想定
    - .ckpt: Lightning 形式（state_dict を持つ）を想定
      さらに ema_shadow が入っていれば、use_ema_infer=True のとき EMA を適用する
    """
    obj: Any = torch.load(weight_path, map_location="cpu")

    sd: Optional[Dict[str, torch.Tensor]] = None
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        sd = _strip_prefix(sd, prefixes=["model.", "net.", "module."])
        if "ema_shadow" in obj and isinstance(obj["ema_shadow"], dict):
            ema_shadow = obj["ema_shadow"]
    elif isinstance(obj, dict):
        # .pth で state_dict を直に保存している場合
        sd = obj
        sd = _strip_prefix(sd, prefixes=["model.", "net.", "module."])
        # もし .pth に ema_shadow も入れていた場合に備える
        if "ema_shadow" in obj and isinstance(obj["ema_shadow"], dict):
            ema_shadow = obj["ema_shadow"]
    else:
        raise ValueError(f"Unsupported weight format: {weight_path}")

    assert sd is not None
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]} ... (total {len(unexpected)})")
    if missing:
        print(f"[warn] missing keys: {missing[:5]} ... (total {len(missing)})")

    # --- 追加：推論で EMA を適用 ---
    if use_ema_infer and (ema_shadow is not None):
        _apply_ema_shadow_to_model(model, ema_shadow)
        print("[info] EMA shadow applied for inference.")
    elif use_ema_infer and (ema_shadow is None):
        print("[info] EMA shadow not found in checkpoint. Use raw weights for inference.")

    return model


# =====================
# 推論
# =====================
@torch.no_grad()
def predict_ensemble(
    ckpt_paths: List[Path],
    model_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    test_img: pd.DataFrame,
    input_dir: Path,
    device: torch.device,
    use_ema_infer: bool = True,
) -> np.ndarray:
    tfm = create_transforms(image_size)
    ds = ImageDataset(df=test_img, input_dir=input_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pred_sum: Optional[np.ndarray] = None

    for p in ckpt_paths:
        model = create_model(model_name=model_name, num_classes=3)
        model = load_weights_flexible(model, p, use_ema_infer=use_ema_infer)
        model.to(device)
        model.eval()

        preds_all = []
        for images in loader:
            images = images.to(device, non_blocking=True)
            preds = model(images).float()  # (B,3)
            preds_all.append(preds.detach().cpu().numpy().astype(np.float32))

        pred3 = np.concatenate(preds_all, axis=0)  # (N,3)
        pred_sum = pred3 if pred_sum is None else (pred_sum + pred3)

    pred3_mean = pred_sum / max(1, len(ckpt_paths))
    return pred3_mean


# =====================
# submission 作成（sample_submission.csv 不要）
# =====================
def build_submission_from_testcsv(
    pred3: np.ndarray,
    test_img: pd.DataFrame,
    test_long: pd.DataFrame,
) -> pd.DataFrame:
    if pred3.shape[1] != 3:
        raise ValueError(f"pred3 の形が想定外です: {pred3.shape} （(N,3) を想定）")

    # 提出csv作成時はマイナス値を0にクリップ
    pred3 = np.clip(pred3, 0.0, None).astype(np.float32)

    # 3出力（あなたの順序に合わせる）
    pred_green = pred3[:, 0]
    pred_clover = pred3[:, 1]
    pred_dead = pred3[:, 2]

    # 3 -> 5 復元（あなたのロジック）
    pred_gdm = pred_green + pred_clover
    pred_total = pred_green + pred_clover + pred_dead

    pred_wide = test_img.copy()
    pred_wide["Dry_Green_g"] = pred_green
    pred_wide["Dry_Clover_g"] = pred_clover
    pred_wide["Dry_Dead_g"] = pred_dead
    pred_wide["GDM_g"] = pred_gdm
    pred_wide["Dry_Total_g"] = pred_total

    # test.csv の行（=提出行）をそのまま使う
    sub = test_long[["sample_id", "sample_id_suffix", "image_path"]].copy()
    sub = sub.merge(pred_wide[["image_path"] + TARGET5_ORDER], on="image_path", how="left")

    # suffix に応じて target を選ぶ
    sub["target"] = 0.0
    for col in TARGET5_ORDER:
        m = sub["sample_id_suffix"].astype(str) == col
        sub.loc[m, "target"] = sub.loc[m, col].astype(float)

    # 取りこぼし検知
    if sub["target"].isna().any():
        bad = sub[sub["target"].isna()][["sample_id", "sample_id_suffix"]].head(10)
        raise ValueError(f"target が NaN になりました。suffix が想定外です。例:\n{bad}")

    return sub[["sample_id", "target"]]


# =====================
# main
# =====================
def main():
    # ===== ここを自分の環境に合わせて変更 =====
    COMP_DIR = Path("/kaggle/input/csiro-biomass")  # 競技データのディレクトリ
    WEIGHTS_DIR = Path("/kaggle/input/hoge")        # 重みデータセット名に置換

    CKPTS = [
        WEIGHTS_DIR / "model_fold0-best.ckpt",
        WEIGHTS_DIR / "model_fold1-best.ckpt",
        WEIGHTS_DIR / "model_fold2-best.ckpt",
        WEIGHTS_DIR / "model_fold3-best.ckpt",
        WEIGHTS_DIR / "model_fold4-best.ckpt",
    ]

    MODEL_NAME = "efficientnet_b2"
    IMAGE_SIZE = 260
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # 追加：推論で EMA shadow を使うか
    USE_EMA_INFER = True
    # ==========================================

    test_csv = COMP_DIR / "test.csv"
    input_dir = COMP_DIR  # image_path が "test/xxx.jpg" 形式ならこれでOK

    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv が見つかりません: {test_csv}")

    for p in CKPTS:
        if not p.exists():
            raise FileNotFoundError(f"weight が見つかりません: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    test_img, test_long = make_test_tables(test_csv)

    pred3 = predict_ensemble(
        ckpt_paths=CKPTS,
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_img=test_img,
        input_dir=input_dir,
        device=device,
        use_ema_infer=USE_EMA_INFER,
    )

    sub = build_submission_from_testcsv(pred3, test_img, test_long)
    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"saved: {out_path}  rows={len(sub)}")
    print(sub.head())


if __name__ == "__main__":
    main()
