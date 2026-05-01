"""Download a subset of the **Chars74K** dataset and flatten it into
``data/chars74k/`` so the project's ``source: dir`` data spec can pick
it up directly.

Why this script exists
----------------------
There is no Hugging Face mirror of Chars74K -- the canonical source is
Google Drive links on the authors' page. We use ``gdown`` to fetch the
chosen subset, untar it, then copy / shuffle the character PNGs into a
single flat directory.

Available subsets (English script, 62 classes 0-9 / A-Z / a-z):

    fnt   ~51 MB   computer-rendered fonts (cleanest upright; ~63K imgs)
    img   ~128 MB  characters cropped from natural scenes (~7.7K imgs)
    hnd   ~13 MB   tablet-drawn handwritten characters    (~3.4K imgs)

Default is ``fnt`` because every image in the synthesized-fonts subset
is *guaranteed upright* by construction -- exactly what you want when
the agent has to learn that "upright" is class 0 of the rotation MDP.
``img`` is closer to the original benchmark spirit but contains
characters that are naturally tilted in the source photo (which makes
"upright" fuzzy and adds noise to the reward signal).

Usage
-----
    pip install gdown
    python scripts/download_chars74k.py --subset fnt --max_images 200

Then in ``configs/combined.yaml`` use:

    data:
      combined:
        - {source: torchvision, name: cifar10, split: train, num_images: 100}
        - {source: dir,         path: data/chars74k, num_images: 100}

Manual fallback
---------------
If Google Drive blocks ``gdown`` (rate limit / quota), download the
tarball yourself from:

    https://teodecampos.github.io/chars74k/

then re-run with ``--tarball /path/to/EnglishFnt.tgz`` and the script
will skip the download step.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List


# Drive file IDs taken from the canonical page maintained by the author.
DRIVE_IDS = {
    "fnt": "1j3aQTIgmoYXycjBSNwAFQ68pChFY1AgV",   # EnglishFnt.tgz, ~51 MB
    "img": "1VNMJEvx9UJwzVi7IaNF2QfbMbAD90krX",   # EnglishImg.tgz, ~128 MB
    "hnd": "1FCH2jjo9Z1HbPVDWAEfvE3ZKaPpaXogg",   # EnglishHnd.tgz, ~13 MB
}


def _gdown_or_explain(url: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        print(
            "[chars74k] ERROR: gdown is required to fetch from Google Drive.\n"
            "           Install with:  pip install gdown\n"
            "           Or pre-download the tarball manually from\n"
            "             https://teodecampos.github.io/chars74k/\n"
            "           and re-run with  --tarball /path/to/Englishxxx.tgz"
        )
        sys.exit(1)

    print(f"[chars74k] downloading -> {dest}")
    gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)


def _extract_pngs(tarball: Path, extract_dir: Path) -> List[Path]:
    """Extract a Chars74K tarball and return the list of character PNGs."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[chars74k] extracting {tarball} -> {extract_dir}")
    with tarfile.open(tarball) as tf:
        tf.extractall(extract_dir)

    candidates = [
        p for p in extract_dir.rglob("*.png")
        if "mask" not in p.name.lower()
        and "binary" not in p.name.lower()
    ]
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subset", choices=sorted(DRIVE_IDS.keys()), default="fnt",
        help="Which Chars74K English subset to fetch (default: fnt).",
    )
    parser.add_argument(
        "--output", default="data/chars74k",
        help="Final flat directory of character PNGs.",
    )
    parser.add_argument(
        "--max_images", type=int, default=200,
        help="How many shuffled character images to keep on disk (default 200; 0 = keep all).",
    )
    parser.add_argument(
        "--tarball", type=str, default=None,
        help="Path to a pre-downloaded .tgz; skips the Google Drive download step.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the cross-class shuffle.",
    )
    parser.add_argument(
        "--keep_tarball", action="store_true",
        help="Keep the downloaded tarball after extraction (default deletes it).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download / re-flatten even if data already exists in --output.",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(out.glob("chars74k_*.png"))
    if existing and not args.force:
        print(
            f"[chars74k] {len(existing)} images already present in {out}; "
            f"skipping (re-run with --force to redo)."
        )
        return

    # Wipe any stale flattened files (but only the ones we ourselves created).
    for p in existing:
        p.unlink()

    if args.tarball is not None:
        tarball = Path(args.tarball).expanduser().resolve()
        if not tarball.exists():
            print(f"[chars74k] ERROR: --tarball not found: {tarball}")
            sys.exit(1)
        downloaded_here = False
    else:
        tarball = out / f"English{args.subset.capitalize()}.tgz"
        downloaded_here = True
        url = f"https://drive.google.com/uc?id={DRIVE_IDS[args.subset]}"
        _gdown_or_explain(url, tarball)

    extract_dir = out / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    pngs = _extract_pngs(tarball, extract_dir)

    if not pngs:
        print(
            "[chars74k] ERROR: no .png files found after extraction.\n"
            "           The tarball layout may have changed -- inspect "
            f"{extract_dir}"
        )
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(pngs)            # shuffle so we get a diverse cross-class mix
    if args.max_images and args.max_images > 0:
        pngs = pngs[: args.max_images]

    print(f"[chars74k] copying {len(pngs)} files to {out}/ ...")
    for i, src in enumerate(pngs):
        dst = out / f"chars74k_{i:05d}.png"
        shutil.copy(src, dst)

    shutil.rmtree(extract_dir, ignore_errors=True)
    if downloaded_here and not args.keep_tarball:
        tarball.unlink(missing_ok=True)

    print(f"[chars74k] done. {len(pngs)} PNGs at {out}/")
    print(
        "[chars74k] now point your config at this directory. Example:\n"
        "    data:\n"
        "      combined:\n"
        "        - {source: torchvision, name: cifar10, split: train, num_images: 100}\n"
        f"        - {{source: dir, path: {out.as_posix()}, num_images: 100}}"
    )


if __name__ == "__main__":
    main()
