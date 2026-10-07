#!/usr/bin/env python3
"""Animate one case's six-class ROI overlay as NCCT / CTP / DWI side by side.

Input is the three QC montages written by generate_6class_visualizations.py,
one per modality, each a grid of axial slices with a z caption per tile and a
six-class legend along the bottom. Tiles are cut back out of that grid, so the
volumes themselves are not needed.

    python make_roi_gif.py --case-dir .../6class_visualizations_cleaned/sub-XX_YYY \
                           --out roi.gif

The CTP slab is shorter than the NCCT and DWI field of view, so the animation is
limited to the slices where CTP actually carries signal; all three modalities
stay in step. No patient identifier is drawn: only the modality names and the
six class labels. The montage title, which does carry the subject id, is never
sampled.
"""

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# montage grid geometry, in pixels of the source montage
CX0, CPX = 130.0, 297.50          # tile centre x = CX0 + CPX * column
RY0, RPY = 146.0, 267.60          # cell top y    = RY0 + RPY * row
CAPTION_H, N_COLS = 42, 10
LEGEND_Y = (1812, 1888)           # the six-class key, well below the title

#: Exact legend colours. Median-cut allocates palette slots by pixel count, so
#: these few-hundred-pixel swatches never earn one and the class colours drift
#: (red to maroon, orange to tan). They are reserved in the palette instead.
CLASS_RGB = [(255, 0, 0), (0, 0, 255), (255, 165, 0),
             (144, 238, 144), (0, 128, 0), (255, 255, 0)]
MODALITIES = ("ncct", "ctp", "dwi")


def cell_box(z):
    row, col = divmod(z, N_COLS)
    y0 = int(round(RY0 + RPY * row)) + CAPTION_H
    y1 = int(round(RY0 + RPY * (row + 1)))
    return CX0 + CPX * col, y0, y1


HALF_W = 150


def pad(montage):
    """Pad left/right so a column-0 tile does not index past the array start.

    The tile window is centred on cx, and cx for column 0 is 130, so cx-150 is
    negative; numpy reads that as an offset from the right edge and returns an
    empty slice, which rendered as a black frame.
    """
    return np.pad(montage, ((0, 0), (HALF_W, HALF_W), (0, 0)))


def cell(montage, z):
    """Tile for slice z out of an already-padded montage."""
    cx, y0, y1 = cell_box(z)
    x0 = int(cx) + HALF_W - HALF_W
    return montage[y0:y1, x0:x0 + 2 * HALF_W]


def grey_fraction(montage, z):
    """Share of pixels that are greyscale anatomy rather than colour overlay."""
    tile = cell(montage, z).astype(int)
    if tile.size == 0:
        return 0.0
    spread = tile.max(2) - tile.min(2)
    return float(((spread < 26) & (tile.max(2) > 28)).mean())


def load_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build(case_dir, out_path, scale, duration, min_coverage, palette_size):
    case_dir = pathlib.Path(case_dir)
    stem = case_dir.name
    montage = {m: pad(np.asarray(Image.open(case_dir / f"{stem}_6class_{m}.png").convert("RGB")))
               for m in MODALITIES}

    slices = [z for z in range(N_COLS * 6)
              if grey_fraction(montage["ctp"], z) >= min_coverage]
    if not slices:
        raise SystemExit("no slice has CTP coverage; check --min-coverage")
    z0, z1 = min(slices), max(slices)
    print(f"  CTP coverage: z={z0}..{z1} ({z1 - z0 + 1} slices)")

    # one crop window that holds every brain across the range and all modalities
    left = right = top = bottom = None
    for z in range(z0, z1 + 1):
        for m in montage.values():
            mask = cell(m, z).sum(2) > 24
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            left = xs.min() if left is None else min(left, xs.min())
            right = xs.max() if right is None else max(right, xs.max())
            top = ys.min() if top is None else min(top, ys.min())
            bottom = ys.max() if bottom is None else max(bottom, ys.max())
    bbox_pad = 6
    left, top = max(0, left - bbox_pad), max(0, top - bbox_pad)
    right, bottom = right + bbox_pad, bottom + bbox_pad

    def tile(m, z):
        return Image.fromarray(cell(m, z)[top:bottom + 1, left:right + 1])

    legend_strip = montage["ncct"][LEGEND_Y[0]:LEGEND_Y[1], HALF_W:-HALF_W]
    xs = np.where((legend_strip.sum(2) > 24).any(0))[0]
    legend = Image.fromarray(legend_strip[:, max(0, xs.min() - 10):xs.max() + 11])

    pw = int((right - left + 1) * scale)
    ph = int((bottom - top + 1) * scale)
    gap, margin, title_h = 14, 16, 30
    width = margin * 2 + pw * 3 + gap * 2
    if legend.width > width - margin * 2:
        k = (width - margin * 2) / legend.width
        legend = legend.resize((int(legend.width * k), int(legend.height * k)), Image.LANCZOS)
    height = margin + title_h + ph + 10 + legend.height + margin
    font = load_font(19)

    def frame(z):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i, m in enumerate(MODALITIES):
            x = margin + i * (pw + gap)
            img.paste(tile(montage[m], z).resize((pw, ph), Image.LANCZOS), (x, margin + title_h))
            label = m.upper()
            draw.text((x + (pw - draw.textlength(label, font=font)) / 2, margin + 4),
                      label, font=font, fill=(236, 240, 242))
        img.paste(legend, ((width - legend.width) // 2, margin + title_h + ph + 10))
        return img

    frames = [frame(z) for z in range(z0, z1 + 1)]

    sample = Image.new("RGB", (width, height * 4))
    for i, f in enumerate(frames[::max(1, len(frames) // 4)][:4]):
        sample.paste(f, (0, i * height))
    n_free = palette_size - len(CLASS_RGB) - 2
    base = sample.quantize(colors=n_free, method=Image.MEDIANCUT)
    entries = CLASS_RGB + [(0, 0, 0), (255, 255, 255)] + \
              [tuple(base.getpalette()[i * 3:i * 3 + 3]) for i in range(n_free)]
    palette = Image.new("P", (1, 1))
    palette.putpalette([v for rgb in (entries + [(0, 0, 0)] * 256)[:256] for v in rgb])

    quantised = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(out_path, save_all=True, append_images=quantised[1:],
                      duration=duration, loop=0, optimize=True, disposal=1)
    print(f"  wrote {out_path}  {len(frames)} frames, {width}x{height}, "
          f"{out_path.stat().st_size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case-dir", required=True,
                    help="a 6class_visualizations_cleaned/<case> directory")
    ap.add_argument("--out", default="roi.gif")
    ap.add_argument("--scale", type=float, default=1.35)
    ap.add_argument("--duration", type=int, default=110, help="ms per frame")
    ap.add_argument("--min-coverage", type=float, default=0.05,
                    help="minimum grey-anatomy fraction for a CTP slice to count")
    ap.add_argument("--palette-size", type=int, default=160)
    args = ap.parse_args()
    build(args.case_dir, args.out, args.scale, args.duration,
          args.min_coverage, args.palette_size)


if __name__ == "__main__":
    main()
