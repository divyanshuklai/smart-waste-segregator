# Smart Waste Segregator

A smart bin that detects incoming trash and drops it into one of four bins: **organic, plastic, metal, glass**.

A camera on the bin takes a photo when something is thrown in. A small image classifier runs on a Raspberry Pi and picks one of the four classes. Two servo-driven flaps then route the item to the right compartment.

This repo holds two generations of the work:

- `rpi-mobilenet/` is the version that ran on the bin (November 2024). PyTorch training, ONNX export, onnxruntime on the Pi.
- `yolo11-exports/` is a later look at YOLO11 for edge deployment (March 2025). TorchScript and NCNN exports, no waste-specific training.

The honest results are in [Limitations](#limitations--what-went-wrong). The bin worked. The model did not work well on real trash.

## Timeline

The project predates the author using git, so there is no commit history from the time.
What follows is reconstructed from filesystem modification times on the surviving
backup folders. Treat it as evidence rather than proof: mtimes survive a copy but can
be rewritten by one, and the folders were moved more than once before landing here.

| Date | What happened |
|---|---|
| 2024-10-11 | First code: `camera-test.py` (OpenCV capture attempt) and `exporter-v2-early.py` (a two-class MobileNetV2 export). Both are dead ends, both are kept below. |
| 2024-11-06 | Training day. `sws.ipynb` 16:23, `waste_seg.ipynb` 17:44, then `best_model.pth` 18:41, `export.py` 18:43 and `waste_classifier.onnx` 18:44 within three minutes of each other. |
| 2024-11-07 | Pi integration, 02:33 to 10:21. `pir_test.py`, the three test JPEGs, then `smart_waste_segregation.py` at 10:17 and `inference.py` at 10:21. This is the night the `libcamera-still` capture path was written. |
| 2024-11-13 | The evaluation attempt: `confusion_matrix.png`, `error_log.txt` and `prediction_results.csv`, all written at 17:05. |
| 2024-11-23 | Final notebook edits to `sws.ipynb`, `test.ipynb` and `waste_seg.ipynb`. |
| 2025-03-02/03 | Generation 2, a year later: the YOLO11 TorchScript and NCNN exports in `yolo11-exports/`. |

The four weeks between the first camera attempt on 11 October and the working capture
path on 7 November are the hardware troubleshooting described under
[Limitations](#limitations--what-went-wrong). That gap is the honest shape of the
project: most of the calendar went to getting hardware to cooperate, not to the model.

## Hardware

- Raspberry Pi as the only compute. No accelerator, no offboard inference.
- PiCamera, driven through `libcamera-still` at 2592x1944.
- LIDAR sensor as the trigger. The Pi wakes, takes one photo, classifies it, then goes back to idle.
- PIR motion sensor, used in a test rig on BCM pin 17 (`rpi-mobilenet/pir_test.py`). The PIR needs a 60 second settle before it reports anything useful.
- Two MG995 metal-gear servos driving a dual-flap mechanism. The dual-flap design was compact and the metal-gear servos held up to repeated actuation.

The LIDAR trigger is the design decision that mattered most. The first plan was to process a continuous video stream and detect the object in frame. On a Pi that pins the CPU and the board overheats. A LIDAR trip that fires a single capture removed the idle load completely, and the thermal problem went away with it.

### Getting a picture out of the camera

The Python camera libraries could not see the camera. The operating system could:
the device enumerated, and `libcamera-still` on the command line produced a JPEG
every time. Only the bindings were blind.

`camera-test.py` is the first attempt, dated 11 October 2024. It is four lines of
OpenCV and it fails at the second one:

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()
```

The fix, in `smart_waste_segregation.py` four weeks later, is to stop fighting the
bindings and drive the vendor's own tool instead, writing to a file and reading it
back with OpenCV:

```python
command = [
    'libcamera-still',
    '-o', image_path,
    '--nopreview',
    '--width', '2592',
    '--height', '1944',
    '-t', '1000'
]
subprocess.run(command, check=True)
```

This is worth stating plainly because it looks like a workaround and is not one.
`libcamera-still` is the supported Raspberry Pi capture tool; the `libcamera` stack had
replaced the legacy camera interface, and the Python bindings on the OS image of the
day did not reliably reach it. Shelling out to the vendor CLI and reading the file back
is the intended path when the bindings are the broken layer, and it costs one process
spawn per capture on a pipeline that already budgets three seconds end to end.

The capture is invoked with an argument list rather than a shell string, so filenames
carrying spaces or shell metacharacters cannot break it, and `check=True` turns a
failed capture into a `CalledProcessError` that the caller handles rather than a
silently missing file that surfaces later as a confusing OpenCV `None`.


## Pipeline (generation 1)

Trigger, capture, classify, actuate. Total time under 3 seconds, which came from picking a small MobileNet-family classifier rather than something heavier.

Preprocessing, identical in `inference.py` and `smart_waste_segregation.py`:

1. Read the JPEG with OpenCV.
2. BGR to RGB.
3. Resize to 224x224.
4. Scale to `[0, 1]`, then ImageNet normalization, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
5. Transpose HWC to NCHW, add a batch axis, cast to float32.

Model: `torchvision` MobileNetV3-Small, ImageNet weights, last classifier layer replaced with a 4-way `Linear`. Exported to ONNX at opset 11 with `do_constant_folding=True`, input named `input` and output named `output`, batch axis dynamic. Input shape `(1, 3, 224, 224)`, output shape `(1, 4)`. The exported file `waste_classifier.onnx` is 6.1 MB. The PyTorch version recorded in the ONNX file is 2.4.1.

On-device session config, from `smart_waste_segregation.py`:

```python
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4

session = onnxruntime.InferenceSession(
    model_path,
    providers=['CPUExecutionProvider'],
    sess_options=session_options,
)
```

Softmax is applied by hand on the raw logits, with a max subtraction for numerical stability. The script reports the predicted class, the confidence, and the measured inference time.

### Training runs

Two training notebooks are here, and they are not the same experiment.

`waste_seg.ipynb` is the run that produced the shipped weights. MobileNetV3-Small, Adam at lr 0.001, 224x224 inputs, class grouping:

| Index | Class | Source folders |
|---|---|---|
| 0 | metal | `metal` |
| 1 | plastic | `plastic` |
| 2 | glass | `brown-glass`, `green-glass`, `white-glass` |
| 3 | biodegradable | `biological`, `paper`, `cardboard` |

6571 images total, split 5256 train and 1315 validation. Ten epochs, best validation accuracy 91.63% at epoch 9. A second, longer run in the same notebook adds heavier augmentation (random resized crop, flips, colour jitter, Gaussian blur, rotation) with `OneCycleLR`, early stopping patience 5, 20 epochs over 165 batches. Its per-epoch numbers were not kept in the notebook output.

`sws.ipynb` is a separate MobileNetV2 experiment on a larger merged dataset: 7324 images from a YOLO-format waste set plus 15515 from a folder-per-class set, mapped down to `['organic', 'plastic', 'metal', 'glass']` and split 18271 train / 4568 validation. A `WeightedRandomSampler` was used because the merged data is badly skewed:

| Class | Images | Share |
|---|---|---|
| organic | 14707 | 64.4% |
| glass | 3863 | 16.9% |
| metal | 2588 | 11.3% |
| plastic | 1681 | 7.4% |

That run did not produce the deployed checkpoint. `mobilenetv3-small/best_model.pth` on Hugging Face is the MobileNetV3-Small state dict from `waste_seg.ipynb`, and `waste_classifier.onnx` was exported from it. The ONNX graph contains squeeze-and-excite blocks and `HardSigmoid` nodes, which confirms V3 rather than V2.

Note the class order is inconsistent across the files in this repo. `waste_seg.ipynb` and `inference.py` use `['metal', 'plastic', 'glass', 'biodegradable']`, which matches the shipped weights. `sws.ipynb` uses `['organic', 'plastic', 'metal', 'glass']`, and `smart_waste_segregation.py` has a third ordering, `['plastic', 'paper', 'metal', 'glass']`, which is wrong for the model it loads. The strings printed by the Pi script are therefore mislabelled even when the argmax is right. It is left as-is because it is part of what went wrong.

### Evaluation

`test.ipynb` ran the ONNX model against a held-out YOLO-format set and wrote `confusion_matrix.png`, `prediction_results.csv` and `error_log.txt`.

![Confusion matrix from test.ipynb](rpi-mobilenet/confusion_matrix.png)

The numbers it printed are 2.75% accuracy over 182 images, with 860 of 1042 images skipped. Do not read that as the model's accuracy. The evaluation itself is broken:

- The label files come from a six-class YOLO set (`BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`). `test.ipynb` reads the raw class index and rejects anything at or above 4, so every `PAPER` and `PLASTIC` image was dropped. That is the 860 failures in `error_log.txt`.
- The four indices that survive are then read against a different four-class list, so a `METAL` image is scored as `biodegradable`. `prediction_results.csv` shows this directly: `metal1000_jpg....jpg` has `true_class` recorded as `biodegradable`.

So the 150 biodegradable-predicted-metal cell in the confusion matrix is mostly a label mapping bug, not a model failure. The real conclusion is weaker and worse: there was never a valid held-out test set for this model.

## Generation 2: YOLO11 exports

`yolo11-exports/model.ipynb` is exploratory. It takes the stock Ultralytics `yolo11n` detection and `yolo11n-cls` classification checkpoints and exports each to TorchScript and to NCNN, which is the format worth using on a Pi-class CPU.

- Ultralytics 8.3.82, torch 2.6.0+cu118, CPU export.
- `yolo11n-cls`: 47 layers, 2,807,024 parameters, 4.2 GFLOPs, input `(1, 3, 224, 224)`, output `(1, 1000)`. TorchScript 10.9 MB, NCNN 10.7 MB.
- `yolo11n`: COCO detection, 80 classes, input 640x640, stride 32.
- NCNN export runs through `pnnx` and produces `model.ncnn.param`, `model.ncnn.bin` and a generated `model_ncnn.py`.
- A sanity check ran the NCNN classifier on `bus.jpg`: 124.0 ms inference, 78.8 ms preprocess, 1.3 ms postprocess.

These are the stock ImageNet and COCO weights. No waste dataset was trained here, and no NCNN model was ever deployed to the bin. This generation is export plumbing and a latency measurement, nothing more.

## Weights

No weights are committed here. They live on Hugging Face:

**https://huggingface.co/TheHelltaker/smart-waste-segregator**

```
mobilenetv3-small/
  best_model.pth            18.5 MB   trained state dict
  waste_classifier.onnx      6.1 MB   ONNX export, this is what runs on the Pi
yolo11/
  yolo11n.pt, yolo11n-cls.pt          stock Ultralytics checkpoints
  *.torchscript                       TorchScript exports
  yolo11n_ncnn_model/                 NCNN export
  yolo11n-cls_ncnn_model/             NCNN export
```

Fetch them with:

```bash
pip install -U huggingface_hub
hf download TheHelltaker/smart-waste-segregator --local-dir weights
```

## Repo layout

```
rpi-mobilenet/                    generation 1, the version that ran on the bin
  smart_waste_segregation.py      Pi runtime: libcamera-still capture, ONNX inference
  inference.py                    standalone classifier class, single image
  export.py                       PyTorch to ONNX, opset 11, with a parity check
  pir_test.py                     PIR sensor test rig, BCM pin 17
  waste_seg.ipynb                 MobileNetV3-Small training, produced the shipped weights
  sws.ipynb                       separate MobileNetV2 run on a larger merged dataset
  test.ipynb                      ONNX evaluation, wrote the artifacts below
  confusion_matrix.png            evaluation output, see the caveat above
  prediction_results.csv          per-image predictions, 182 rows
  error_log.txt                   860 skipped images from the broken label mapping
  camera-test.py                  first camera attempt, OpenCV, did not work
  exporter-v2-early.py            earliest surviving code, two-class MobileNetV2
  metal_test.jpg                  test images used during bring-up
  pet_test.jpg
  plastic_test.jpg

yolo11-exports/                   generation 2, YOLO11 edge export experiments
  model.ipynb                     export runs and the NCNN sanity check
  bus.jpg                         Ultralytics sample images
  zidane.jpg
  pyproject.toml                  uv project, Python 3.13
  uv.lock
  main.py                         stub
```

## Running it

### On the Pi

Needs `onnxruntime`, `opencv-python`, `numpy`, and `libcamera-still` on the path.

```bash
cd rpi-mobilenet
python smart_waste_segregation.py
```

It counts down 5 seconds, captures at 2592x1944, classifies, and prints the class, confidence and inference time. Point it at the downloaded `waste_classifier.onnx` by editing `model_path`, and fix `class_names` to `['metal', 'plastic', 'glass', 'biodegradable']` before trusting the label it prints.

### On a single image, off the Pi

```bash
cd rpi-mobilenet
python inference.py
```

Defaults to `waste_classifier.onnx` and `metal_test.jpg`. Adjust the paths at the top of `main()`.

### PIR test rig

Needs `RPi.GPIO`. Wire the sensor to BCM 17 and give it a minute to settle.

```bash
python rpi-mobilenet/pir_test.py
```

### Re-export the ONNX model

```bash
cd rpi-mobilenet
python export.py
```

It loads `best_model.pth`, exports to `waste_classifier.onnx`, runs `onnx.checker`, and compares the ONNX output against PyTorch at `rtol=1e-3, atol=1e-5`. Paths at the top of `convert_to_onnx()` assume both files sit beside the script.

### YOLO11 exports

```bash
cd yolo11-exports
uv sync
uv run python -c "from ultralytics import YOLO; YOLO('yolo11n-cls.pt').export(format='ncnn')"
```

`ultralytics` is not in `pyproject.toml`, so add it first. The committed exports were produced with 8.3.82.

## Limitations / what went wrong

This is the part worth reading.

**The dataset was biased and the model did not survive contact with real trash.** Training used public waste datasets. Those images are clean, centred, well lit, one object per frame. Real trash going into a bin is crumpled, dirty, partly occluded and photographed from above under whatever light the room has. Validation accuracy above 90% did not translate. Real-world performance was poor.

**Online waste datasets do not look like waste.** This is the same problem stated from the data side, and it is the root cause. Nobody collected images from the actual bin with the actual camera at the actual angle. A few hundred images photographed through the deployed PiCamera would have been worth more than the tens of thousands of scraped images that were used.

**There was no proper held-out test set.** The only evaluation attempt is `test.ipynb`, and as described above it mapped labels from a six-class set onto a four-class model and silently dropped 860 of 1042 images. Its 2.75% accuracy number is meaningless in both directions. Without a clean test set there was no way to measure whether any change helped, so the model was tuned against validation accuracy on the same biased distribution it was trained on.

**Hardware ate the schedule.** A large share of the project time went to hardware troubleshooting rather than to the model or the data. The PiCamera was the worst of it, between driver and stack changes and getting a reliable capture out of `libcamera-still`. Time spent there is time not spent collecting a real dataset, which is most of why the first failure above never got fixed.

## Model weights and licences

The YOLO11 checkpoints on Hugging Face are the stock Ultralytics releases and carry the AGPL-3.0 licence, as recorded in their `metadata.yaml`. `bus.jpg` and `zidane.jpg` are Ultralytics sample images. The MobileNetV3-Small weights on Hugging Face were trained here, starting from torchvision ImageNet weights.
