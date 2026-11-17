"""
Batch-apply FSE enhancement to a dataset organised as:

    <input_root>/
        train/  *.png
        test/   *.png
        val/    *.png

Results are written to <output_root>/ with the same three subfolders and the
same file names. Uses the FSE implementation in fse.py (Chen et al., 2025).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from scripts.fse import fse, image_std

INPUT_ROOT = "./dataset/syntax/images"
OUTPUT_ROOT = "./dataset/syntax/fse_images"
SPLITS = ("train", "test", "val")
EXTS = (".png",)  # add ".jpg", ".jpeg", ".bmp", ".tif" here if you need them


def process_split(in_dir: Path, out_dir: Path,
                  ksize: int, sigma: float, levels: int,
                  paper_denominator: bool) -> tuple[int, int]:
    """Enhance every image in one split folder. Returns (done, skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)

    done = 0
    skipped = 0
    for f in tqdm(files):
        try:
            img = np.asarray(Image.open(f))
            enhanced = fse(img, ksize=ksize, sigma=sigma, L=levels,
                           paper_denominator=paper_denominator)
            Image.fromarray(enhanced).save(out_dir / f.name)
            done += 1
        except Exception as e:  # keep going even if one file is corrupt
            print(f"  [skip] {f.name}: {e}")
            skipped += 1
    return done, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Batch FSE over train/test/val PNG folders.")
    p.add_argument("--input-root", default=INPUT_ROOT,
                   help=f"folder containing train/test/val (default: {INPUT_ROOT})")
    p.add_argument("--output-root", default=OUTPUT_ROOT,
                   help=f"where the enhanced splits are written (default: {OUTPUT_ROOT})")
    p.add_argument("--ksize", type=int, default=5, help="smoothing window (default 5)")
    p.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (default 1.0)")
    p.add_argument("--levels", type=int, default=256, help="grayscale levels L (default 256)")
    p.add_argument("--std-denominator", action="store_true",
                   help="use canonical (M*N - CDF_min) denominator instead of the paper's (M*N - 1)")
    args = p.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    if not in_root.is_dir():
        raise SystemExit(f"input root not found: {in_root.resolve()}")

    total_done = 0
    for split in SPLITS:
        in_dir = in_root / split
        out_dir = out_root / split
        if not in_dir.is_dir():
            print(f"[warn] missing split, skipping: {in_dir}")
            continue
        done, skipped = process_split(
            in_dir, out_dir,
            ksize=args.ksize, sigma=args.sigma, levels=args.levels,
            paper_denominator=not args.std_denominator,
        )
        total_done += done
        print(f"{split:5s}: {done} enhanced, {skipped} skipped -> {out_dir}")

    print(f"done. {total_done} images written under {out_root.resolve()}")


if __name__ == "__main__":
    main()