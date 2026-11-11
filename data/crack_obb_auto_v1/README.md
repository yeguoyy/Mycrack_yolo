# Crack OBB Auto-Conversion v1

This dataset reuses the source-photo-group assignment in `dev/data/mnw_crack_det/manifest.csv` and lives under `yolo/data/crack_obb_auto_v1/`. Source images and masks remain in `dev/data/` and are not modified. Generated image files are hard links when supported by the filesystem. No files are added to `dev/data_splits/`.

Run `yolo/prepare_crack_obb.py` to regenerate the labels and dataset metadata. The Ultralytics dataset YAML is `yolo/data/crack_obb_auto_v1/data.yaml`; review sheets are in this dataset's `review/` directory.

## Label conversion

- Green mask pixels `(0, 128, 0)` are the single `crack` class. Red spalling pixels `(128, 0, 0)` are not converted into crack boxes.
- Each 8-connected crack component of at least 12 pixels becomes one minimum-area rotated rectangle. Components smaller than 12 pixels are omitted. Degenerate rectangle sides are expanded to at least 2 pixels.
- The four OBB corners are normalized by image width and height and written in Ultralytics OBB format: `class x1 y1 x2 y2 x3 y3 x4 y4`.
- The source split contains 147 source groups, kept wholly within train, validation, or test. `source_groups.csv` and `manifest.csv` record the generated split and conversion counts.

## Limitations

These are automatically derived boxes, not manually verified OBB annotations. One box per connected component can poorly approximate curved or branched cracks, and exact-color connected components can fragment at gaps. Inspect the contact sheets under `review/` before training. In particular, review long, curved, branched, and fragmented cracks; revise the conversion strategy or manually correct labels if the boxes do not represent the intended detection targets. Metrics from this dataset must be described as mask-to-OBB adapted results, not results on native OBB annotations.

The 59 source images with no generated crack boxes remain as negative samples. Some may contain only spalling or components below the 12-pixel filter, so verify them before treating them as confirmed crack-free negatives.

## Using with Ultralytics YOLO

After reviewing/correcting labels, run the Python training script from the project root. It uses the existing `dev/.venv` Ultralytics installation and keeps the dataset and run outputs under `yolo/`:

```powershell
.\dev\.venv\Scripts\python.exe .\yolo\scripts\train_crack_obb.py
```

The resulting best checkpoint is normally `yolo/runs/crack_yolo26n/weights/best.pt`; the resumable checkpoint is `yolo/runs/crack_yolo26n/weights/last.pt`. Each `last.pt` is written after a completed epoch, so shutting down mid-epoch only loses progress from that unfinished epoch. To resume an interrupted run from `last.pt`:

```powershell
.\dev\.venv\Scripts\python.exe .\yolo\scripts\train_crack_obb.py --resume
```

Validate the best checkpoint with:

```powershell
.\dev\.venv\Scripts\yolo.exe obb val model=yolo\runs\crack_yolo26n\weights\best.pt data=yolo\data\crack_obb_auto_v1\data.yaml split=test imgsz=640 device=0
```

Training parameters can be overridden, for example `--epochs 150 --batch 4 --device 0 --name crack_yolo26n_v2`. The script does not start training until explicitly run. Do not train on the current draft until the previewed boxes, especially curved/branched cracks and empty-label images, have been reviewed. Results must be described as mask-to-OBB adapted results, not native human-annotated OBB results.
