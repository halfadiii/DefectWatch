# DefectWatch

Upload a photo of a manufactured part and get back a verdict (good or defective), a heatmap of where it looks wrong, and a box around the worst area. Behind that sits the rest of what a model needs to survive in production: experiment tracking, tests, a container, monitoring for drift, and retraining that waits for a person to approve it.

Built on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), a public set of 15 industrial product types.

**Status:** training, the API and drift monitoring work and are measured on a laptop. The container build, the Cloud Run deployment and the demo page are not done yet.

## Results so far

All figures are on MVTec AD's official test split. Full tables are in [reports/](reports/).

| What | Result | Where it comes from |
|---|---|---|
| Image AUROC, mean of 15 product types | 0.974 | [reports/serving_models.md](reports/serving_models.md) |
| F1, threshold tuned on other images (held out) | 0.949 | same |
| F1, threshold tuned on the same images (how papers report it) | 0.966 | same |
| Good parts wrongly failed, at the deployed thresholds | 46 of 467 | same |
| Defects missed, at the deployed thresholds | 36 of 1,258 | same |
| All 15 models, exported to ONNX | 225 MB | same |
| API latency on a laptop, all cores: median / 95th percentile | 93 ms / 123 ms | [reports/latency_local.json](reports/latency_local.json) |
| API latency on a laptop, one thread: median / 95th percentile | 145 ms / 176 ms | [reports/latency_local_1thread.json](reports/latency_local_1thread.json) |
| Drift alarm, clear change (darken to 70%, or blur by 4 px): images until it fires, median | 15 and 14, caught in 70 of 70 runs | [reports/drift_detection.md](reports/drift_detection.md) |
| Drift alarm, false alarms on clean streams (14 product types) | 18 of 42,420 windows | same |
| Good parts wrongly failed when images are blurred by 4 px | 73%, up from 12% | same |

Not measured yet: container size, cold-start time, latency on Cloud Run.

The weakest product types are toothbrush (AUROC 0.88, only 60 training images) and pill (0.94). They are reported, not hidden.

## How it works

**The model is PatchCore, not a detector like YOLO.** MVTec's training images are all good parts, and its defects are marked as pixel masks, not boxes. A detector needs labelled defects to learn from; there are none to train on. PatchCore instead memorises what small patches of good parts look like. At test time, any patch that is far from everything it has memorised is suspicious. That gives a heatmap, the image score comes from the worst patch, and the box is drawn around the hottest connected area.

**The backbone is ResNet-18 with a 1% memory bank.** Measured against the same model with a 10% memory bank, the 1% version scored the same or better (0.974 against 0.970 mean AUROC) at a tenth of the size. See [reports/accuracy.md](reports/accuracy.md).

**Serving runs on ONNX, with no PyTorch in the container.** `serving/export.py` exports each model, then re-scores the whole test split through ONNX using the service's own image loading. The decision threshold stored with each model is tuned on those scores, so it matches exactly what the service sees.

**Two F1 figures, on purpose.** MVTec has no separate labelled set for choosing a threshold. Tuning the threshold on the test images and then scoring on the same images flatters the result, so the held-out figure tunes on half and scores on the other half.

**Monitoring watches capture conditions, not just scores.** Each prediction is logged with the image's brightness, contrast and sharpness and a 16-number summary of its embedding. The uploaded image itself is never stored. The alarm fires when at least 4 of those 19 numbers have shifted (Kolmogorov-Smirnov test, p below 0.001) in the latest 50 predictions, compared with a stretch of predictions known to be normal. Evidently applies the rule and writes the report.

How strict to make the alarm was measured, not guessed. The first rule tried (3 numbers at p below 0.01) raised the alarm on about 15% of clean windows. [reports/drift_calibration.md](reports/drift_calibration.md) has the table the final rule was chosen from.

**MVTec's own test photos have already drifted from its training photos.** For 7 of the 15 product types, the good test photos differ from the good training photos by enough to raise the alarm (carpet: 11 of the 19 watched numbers), while two random halves of the training photos never do. A monitor that only knows the training photos would raise the alarm on a normal stream. So the reference has to be a normal stretch of the live stream itself.

**Retraining will wait for a person.** When lighting or focus changes, good parts start to look like defects. Retraining automatically on those images would teach the model that whatever is in front of the camera is normal, real defects included. So the design is that an alarm opens a retraining request and stops there until someone approves it. The alarm is built and measured; the approval step comes with the deployment.

## Run it

Needs Python 3.11. Training needs an NVIDIA GPU (6 GB is enough for ResNet-18); everything else runs on CPU.

```
# 1. Training environment
py -3.11 -m venv .venv
.venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\pip install -r requirements-train.txt -r requirements-serve.txt

# 2. Data (5.3 GB download), train, export
.venv\Scripts\python training\get_data.py
.venv\Scripts\python training\train.py --categories all --backbone resnet18 --coreset 0.01
.venv\Scripts\python training\report.py
.venv\Scripts\python serving\export.py --categories all --backbone resnet18 --coreset 0.01
.venv\Scripts\python serving\report.py

# 3. Serve
.venv\Scripts\python -m uvicorn serving.app:app --port 8080
curl -F category=bottle -F file=@datasets/MVTecAD/bottle/test/broken_small/003.png http://127.0.0.1:8080/predict

# 4. Latency
.venv\Scripts\python serving\benchmark.py --url http://127.0.0.1:8080 --categories bottle screw --label local

# 5. Monitoring (its own environment: Evidently does not yet work with pandas 3)
py -3.11 -m venv .venv-monitor
.venv-monitor\Scripts\pip install -r requirements-test.txt
.venv-monitor\Scripts\python monitoring\calibrate.py
.venv-monitor\Scripts\python monitoring\simulate_drift.py --categories all
.venv-monitor\Scripts\python -m pytest tests -q
```

Training runs are recorded in MLflow (a local `mlflow.db` by default; set `MLFLOW_TRACKING_URI` to log elsewhere). View them with `.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Layout

| Folder | What is in it |
|---|---|
| `training/` | Download the data, train PatchCore, score it, record runs in MLflow |
| `serving/` | ONNX export, the FastAPI service, latency benchmark |
| `monitoring/` | The drift rule, its calibration, the scheduled check, the drift simulation |
| `tests/` | 32 tests. The API tests use a tiny stand-in model, so CI needs neither the dataset nor a GPU |
| `reports/` | Generated result tables |
| `models/` | Exported models (not in git; rebuild with `serving/export.py`) |

## Data and licence

MVTec AD is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use, with credit. This project is a non-commercial portfolio piece.

Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. "MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019.

PatchCore: Karsten Roth et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. Implementation from [anomalib](https://github.com/open-edge-platform/anomalib).
