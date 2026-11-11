# fine-crack-tracing

Turns a crack segmentation mask into measured centrelines: where each crack runs, how long it is, how wide it is at every point along it, where it branches, and where the mask broke it.

[![tests](https://github.com/Kronbii/fine-crack-tracing/actions/workflows/tests.yml/badge.svg)](https://github.com/Kronbii/fine-crack-tracing/actions/workflows/tests.yml)
[![release](https://img.shields.io/github/v/release/Kronbii/fine-crack-tracing?sort=semver)](https://github.com/Kronbii/fine-crack-tracing/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![A cracked concrete wall with its crack traced as a line coloured by width, branches marked with rings, and short dashed stretches where breaks in the mask were joined](docs/media/hero.jpg)

*A wall from the Concrete Crack Segmentation Dataset (Özgenel, 2019, CC BY 4.0), traced from its hand-drawn mask. The colour is the crack's width at that point, mint (thin) to magenta (wide). Rings are junctions; dashed white stretches cross breaks in the mask.*

A segmentation model tells you which pixels are crack. An inspection needs the next layer: how long the crack is, how wide it gets, where it branches, and a line you can put on a drawing or compare with last year's. This package is that step. It takes any mask (binary, 0/1 labels, grey, colour, RGBA or a probability map) and returns each crack as a trunk and its branches, with the width measured across the crack at every point.

It is not a crack detector. It starts after segmentation, so any model can feed it.

**Contents:** [Quick start](#quick-start) · [How it works](#how-it-works) · [Results](#results) · [Outputs](#outputs) · [Settings](#settings) · [Scoring and benchmarking](#scoring-and-benchmarking) · [What changed in 1.0](#what-changed-in-10) · [Limits](#limits) · [Contributing](#contributing) · [Citing](#citing)

## Quick start

### Install

Python 3.10 or newer. In a virtual environment, install the latest release:

```bash
python -m venv .venv && source .venv/bin/activate
pip install https://github.com/Kronbii/fine-crack-tracing/releases/download/1.0.0/fine_tracing-1.0.0-py3-none-any.whl
```

Or from source, which also gives you the examples and scripts:

```bash
git clone https://github.com/Kronbii/fine-crack-tracing.git
cd fine-crack-tracing
pip install -e .
```

### Try it

The repository has two walls in [examples/](examples):

```bash
fine-tracing trace examples/masks --photos examples/photos -o results/examples
```

```
wall-120                       2 crack(s)      5535 px long   max width 19.0 px      199 ms
wall-609                       2 crack(s)      5353 px long   max width 16.6 px      194 ms
```

`results/examples/overlays/` then holds each photo with its cracks drawn on, and the rest of the folder the measurements (see [Outputs](#outputs)). [examples/quickstart.py](examples/quickstart.py) does the same from Python and prints each crack.

### Your own masks

Masks and the photos they came from are matched by file name; the photos are only used for the overlays and the corner fallback, so they are optional.

```bash
fine-tracing trace data/masks --photos data/photos -o results/
```

With a known pixel size, everything also comes out in millimetres, and width classes give the length of crack in each band:

```bash
fine-tracing trace data/masks -o results/ --px-size 0.08 --width-classes 0.1,0.3,1
```

### From Python

```python
import cv2
from fine_tracing import trace

mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
photo = cv2.imread("photo.jpg")                 # optional
result = trace(mask, photo, px_size=0.08)       # 0.08 mm per pixel

for crack in result.cracks:
    w = crack.width_stats()
    print(crack.id, crack.pattern, f"{crack.length * 0.08:.1f} mm", f"max width {w['max'] * 0.08:.2f} mm")
    for path in crack.paths:          # trunk first, then branches
        path.points                   # (N, 2) x, y every pixel along the line
        path.width                    # (N,) width across the crack at each point, px

result.to_json("result.json")         # also .to_geojson(), .to_svg(), .rows()
```

The tracing settings in [Settings](#settings) are keyword arguments of `trace()` too.

## How it works

![The tracer at work on a wall: the mask fades in, the skeleton sweeps down the cracks with rings at the junctions, then the finished lines draw along every branch, coloured by width](docs/media/trace-run.gif)

![Four panels: the mask, its one-pixel skeleton with ends and junctions, the graph after pruning and bridging with the trunk and branches, and the final line coloured by width](docs/media/stages.png)

1. **Clean.** Binarise (with an optional hysteresis threshold for probability maps), drop specks, fill pinholes.
2. **Thin.** Skeletonise each crack. Diagonal steps that duplicate a path through a 4-neighbour are dropped first, so staircase corners do not read as junctions; clusters of junction pixels become one node.
3. **Prune.** Cut spurs shorter than 2.5 half-widths of the crack at their junction, shortest first, re-checking each junction as it goes.
4. **Check.** In `auto`, a skeleton whose discs rebuild less than 80% of its mask hands that crack to the corner tracer.
5. **Bridge.** Join facing ends across breaks (up to three crack widths, never less than 30 px), then ends that point into the side of another crack.
6. **Extend.** Carry free ends out to the tip of the mask.
7. **Split.** Take the longest end-to-end run as the trunk, then the longest run off what is already covered, until every edge belongs to one path.
8. **Smooth and centre.** Resample each path every pixel and smooth it along its length, more for wider cracks, whose skeletons wiggle with their ragged edges. Then shift the line to the middle of the mask across the crack, keeping only the slow part of that shift so the edge noise is not copied back in.
9. **Measure.** Width across the crack at every point (junction zones and bridges are left out of the statistics), length, orientation, tortuosity, branches, junctions, loops.

## Results

I can't score a centreline against a real mask, because real datasets label pixels, not lines. So the benchmark runs on synthetic cracks with exact ground truth: meandering, zig-zagging, branching cracks from 1.5 to 32 px wide whose true centreline and width are known at every point, rendered as masks and then spoilt the way segmentation output tends to be (ragged edges, breaks, pinholes, specks). Each tracer sees the same 100 images (768 x 768 px, 25 per preset); `fine-tracing bench` reproduces it, and the full table is in [docs/benchmark.md](docs/benchmark.md).

| tracer | F1 within 3 px | distance to the true line | length error | width error | junctions found / real |
|---|---|---|---|---|---|
| v0.1, ORB, sorted by y | 0.10 | 6.1 px | 82% | – | – |
| v0.1, Shi-Tomasi + MST | 0.20 | 5.6 px | 62% | – | – |
| plain skeleton | 0.94 | 1.5 px | 12.6% | 0.75 px | 90% / 42% |
| **v1 (auto)** | **0.97** | **0.52 px** | **1.9%** | **0.65 px** | **83% / 85%** |

*Averages over the four presets below. Length error is the mean absolute error of each image's total crack length. "Found" is the share of true junctions the tracer reports; "real" is the share of reported junctions that are true. The plain skeleton finds most junctions only because it reports every spur as one.*

![Grouped bars for F1 and length error across clean, ragged, broken and realistic masks: v1 stays near F1 1.0 and within 4% of the true length in every case](docs/media/benchmark.png)

On clean masks v1 sits 0.3 px from the true centreline on average, with length within 0.4% and width within 0.3 px. Ragged edges are where pruning and smoothing earn their place: a plain skeleton overstates length by 21% there, v1 by 0.8%, and only one in eight of the plain skeleton's junctions is real. Broken masks are where bridging matters; what is left there (4% short) is mostly crack the mask erased completely, which no tracer can recover. The v0.1 tracers miss most of every crack: with their default quality threshold they keep a few dozen corners for a whole crack, and on wide cracks those corners sit on the edges.

![A generated concrete-like photo, its damaged mask with the true centreline drawn in, and the trace coloured by width](docs/media/synthetic.jpg)

![Distance from the true centreline and length error against crack width for each tracer, single ragged cracks from 1.5 to 32 px wide](docs/media/width-sweep.png)

Width barely moves the result: across single cracks from 1.5 to 32 px wide, v1's length stays within 2.5% of the truth on average, while the v0.1 tracers drift further from the centre the wider the crack.

### On real photos

On real masks there is no true line to score against, so I ran two checks on all 458 hand-drawn masks of the Concrete Crack Segmentation Dataset (8 and 12 megapixels each):

- **What it finds.** 1,008 cracks; branches in 416 of the 458 images; 126 breaks in the masks bridged. The corner fallback was never needed. Each mask takes a median 0.63 s on one core of a laptop CPU.
- **How steady it is.** I damaged each mask the way a weaker model would leave it (ragged edges, breaks, pinholes, specks) and traced it again. Against the trace of the undamaged mask, the new trace scores a mean F1 of 0.96 within 3 px, sits a median 0.86 px away, and its total length changes by a median 0.74%.

![Three more walls from the dataset with their cracks traced and coloured by width, rings at the junctions](docs/media/real-gallery.jpg)

## Outputs

| file | contents |
|---|---|
| `json/<image>.json` | every crack: length, width statistics, orientation, tortuosity, branches, junctions, ends, and every path's points, widths and bridged stretches |
| `cracks.csv` | one row per crack across all images |
| `geojson/<image>.geojson` | paths as LineStrings and junctions as Points, in image pixels |
| `svg/<image>.svg` | the traced lines, coloured by width, as vectors |
| `overlays/<image>.jpg` | the photo (or the mask) with the lines drawn on it |
| `centrelines/<image>.png` | the lines alone, one pixel wide, for other tools |
| `summary.json` | per-image totals, timing and the settings used |

`--formats json,csv` writes only the ones you name. A crack in the JSON, from the wall-609 example traced with `--px-size 0.16`:

```json
{
  "id": 1,
  "method": "skeleton",
  "pattern": "branched",
  "length_px": 5263.61,
  "length_mm": 842.18,
  "gaps": 3,
  "median_width_mm": 0.81,
  "max_width_mm": 2.65,
  "branches": 16,
  "direction": "vertical",
  "tortuosity": 1.141,
  "paths": [
    {
      "id": 0, "parent": null, "order": 1, "length_px": 2374.9,
      "points": [[688.33, -0.5], [687.51, 0.03], [686.61, 0.33], "..."],
      "width_px": [null, 2.11, 2.98, "..."],
      "bridged": [0, 0, 0, "..."]
    }
  ]
}
```

Every field is described in [docs/output-format.md](docs/output-format.md).

## Settings

The tracing settings are flags on `fine-tracing trace`, keywords of `trace()`, fields of `TraceConfig`, and keys in a JSON file passed with `--config` (flags override the file). `--photos`, `--formats`, `--config`, `-j` and `-q` belong to the command only.

| flag | default | what it does |
|---|---|---|
| `--photos` | none | photo file or folder, matched to masks by file name |
| `--method` | `auto` | `skeleton`, `corners`, or `auto` (skeleton, with corners for any crack whose skeleton does not describe its mask) |
| `--px-size`, `--unit` | none, `mm` | size of one pixel, for physical lengths and widths |
| `--width-classes` | none | width class edges in output units; reports the length in each class |
| `--threshold` | 0.5 | mask threshold, as a fraction of full scale |
| `--hysteresis` | none | `low,high` thresholds for probability maps: weak pixels stay only where they touch strong ones |
| `--min-area` | 20 | drop mask specks smaller than this, px |
| `--prune` | 2.5 | cut side branches shorter than this many half-widths |
| `--max-gap` | auto | longest break to bridge, px (auto: three widths of the two ends, at least 30 px) |
| `--max-angle` | 45 | how far each end may turn to meet the other, degrees |
| `--no-bridge`, `--no-extend`, `--no-recentre` | | switch a stage off |
| `--smoothing` | 2 | smoothing along the line, px (wider cracks get up to one half-width more) |
| `--step` | 1 | spacing of the output points, px |
| `--min-length` | 15 | drop cracks shorter than this, px |
| `--formats` | all | which outputs to write: `overlay,json,csv,geojson,svg,centreline` |
| `--config` | none | JSON file of settings |
| `-j` | 1 | images to trace in parallel |
| `-q` | | no per-image lines |

## Scoring and benchmarking

`fine-tracing eval` scores traces (this tool's JSON, or any centreline images) against reference centrelines (JSON, or drawn centreline images): precision, recall and F1 within a tolerance, mean and 95th-percentile distance, length error, and width error and junction recall when both sides have them.

```bash
fine-tracing eval results/json data/reference --tol 3 -o scores/
```

`fine-tracing synth out/ -n 40 --preset realistic` writes synthetic photos, masks and exact ground truth (presets `clean`, `ragged`, `broken`, `realistic`). `fine-tracing bench -o bench/ -j 8` runs every tracer, v0.1 included, on fresh synthetic cracks and writes the tables and charts above. [CONTRIBUTING.md](CONTRIBUTING.md) has the commands that rebuild every figure here and rerun the real-mask check.

## What changed in 1.0

The first release shipped only my starting point: Shi-Tomasi corners linked by a minimum spanning tree. In practice the approach that worked was to skeletonise the mask and fall back to corners only where a skeleton could not describe the crack. Version 1.0 is a rebuild around that, with everything a skeleton needs to hold up on real masks: spur pruning, gap bridging, true ends, a trunk-and-branch split, width along the crack, length measured on a smoothed line (the pixel staircase overstates diagonal cracks by up to 8%), and the corner tracer fixed to trace down the middle of a wide crack rather than around its outline.

![v0.1 traces the outline of each mask with Shi-Tomasi corners; v1 follows the centreline, finds the branches, bridges the break between the top two pieces and colours the line by width](docs/media/before-after.png)

The full list, with what was fixed, is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- It measures the mask it is given. A mask that is a pixel too wide everywhere gives widths a pixel too wide; the width is only as good as the segmentation.
- Width is in pixels until you give it a pixel size, and a pixel size assumes the camera faces the surface square on. It does not correct perspective.
- Bridging guesses that two facing ends are one crack. Bridged stretches are flagged in every output so they can be counted separately or left out.
- Where cracks cross in an X, the crossing becomes two junctions a little apart, and the trunk-and-branch split there is a choice, not a fact.
- The synthetic benchmark is only as realistic as its generator. The real-mask check measures how steady the trace is, not how right it is.

## Repository layout

```
src/fine_tracing/
    pipeline.py   the tracing pipeline: trace(mask, image, ...)
    graph.py      skeleton graph, pruning, bridging, end extension, trunk and branches
    corners.py    the Shi-Tomasi corner tracer (fallback)
    geometry.py   resampling, smoothing, normals, widths across the crack
    mask.py       reading and cleaning masks
    result.py     TraceResult, Crack, Path; JSON, CSV, GeoJSON and SVG export
    render.py     overlays
    metrics.py    scores against reference centrelines
    synth.py      synthetic cracks with exact ground truth
    bench.py      the benchmark
    legacy.py     the v0.1 tracer, kept for the benchmark
    cli.py        the fine-tracing command
docs/             benchmark table, output format, figures
examples/         two walls from the Concrete Crack Segmentation Dataset, and quickstart.py
scripts/          make_figures.py (the images here), real_check.py (the real-mask check)
tests/            pytest suite
test-frames/      a sample crack mask
```

## Contributing

Masks that trace badly are the most useful bug reports; open an issue with the mask, the command and the version. [CONTRIBUTING.md](CONTRIBUTING.md) covers setting up, testing, and keeping the benchmark honest when the tracer changes.

## Citing

If this helps your work, please cite it. GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff):

```
Kronbi, R. (2026). fine-crack-tracing: crack segmentation masks to measured centrelines (Version 1.0.0) [Computer software]. https://github.com/Kronbii/fine-crack-tracing
```

## Credits

- Real photos and masks: Çağlar Fırat Özgenel, [Concrete Crack Segmentation Dataset](https://data.mendeley.com/datasets/jwsn7tfbrp/1), Mendeley Data, 2019, CC BY 4.0. The photos were taken in buildings at Middle East Technical University. The figures here show traced overlays of a few of them, resized, and [examples/](examples) holds two of them at half size.
- Thinning: Zhang and Suen's algorithm, through scikit-image.

## Links

- Project page: [ramikronbi.com/projects/fine-crack-tracing-toolkit](https://ramikronbi.com/projects/fine-crack-tracing-toolkit)
- Write-up: [Turning segmented cracks into measurable paths](https://ramikronbi.com/writing/turning-segmented-cracks-into-measurable-paths)
- Releases: [github.com/Kronbii/fine-crack-tracing/releases](https://github.com/Kronbii/fine-crack-tracing/releases)

MIT License, see [LICENSE](LICENSE).

Built by Rami Kronbi — [ramikronbi.com](https://ramikronbi.com)

