# Smart Waste Segregator

A smart bin that detects incoming trash and then segregates it into one of four
categories: organic, metal, plastic, and glass.

A camera on the lid takes one photo when a LIDAR sensor is tripped. A MobileNetV3-Small
runs on a Raspberry Pi under onnxruntime and picks a class. Two servo-driven flaps route
the item. Built November 2024; I came back to it in March 2025 for the YOLO11 exports.

## Successes

- LIDAR-based detection system proved more efficient than continuous video processing,
  preventing Raspberry Pi overheating.
- Robust MG995 metal servos handled the segregation mechanism effectively.
- The dual-flap design was compact and efficient.
- Reduced latency to under 3 seconds by using a smaller model (MobileNetV3-Small),
  trained to 91.63% validation accuracy and exported to ONNX at opset 11.

## Failures

- The model's real-world performance was poor due to a biased dataset.
- Online training datasets did not reflect real-world waste. I trained on centred
  catalogue photos and deployed on a fixed overhead camera looking at crumpled trash.
- Lack of a proper test set hindered real-world performance evaluation.
- Significant time was lost troubleshooting hardware issues (e.g. PiCamera).
- The 2.75% in `test.ipynb` is a broken label mapping, not a result — six-class labels
  read against a four-class head, dropping 860 of 1042 images. `rescore_test.py` walks
  through it and re-scores the survivors off their filenames.

## Layout

- `rpi-mobilenet/` — the version that ran on the bin. Training notebooks, ONNX export,
  Pi runtime, and the evaluation I got wrong.
- `yolo11-exports/` — TorchScript and NCNN exports of the stock Ultralytics YOLO11n.
  Export plumbing and one latency number; nothing here is trained on waste.

Weights are on Hugging Face, not in the repo:
**https://huggingface.co/TheHelltaker/smart-waste-segregator**

```bash
pip install -U huggingface_hub
hf download TheHelltaker/smart-waste-segregator --local-dir weights
```

## Running it

On the Pi, with `onnxruntime`, `opencv-python`, `numpy` and `libcamera-still` available:

```bash
cd rpi-mobilenet
python smart_waste_segregation.py     # capture, classify, actuate
python inference.py                   # one image, off the Pi
python export.py                      # re-export the ONNX model, with a parity check
python rescore_test.py                # re-score the broken evaluation
```

`class_names` in `smart_waste_segregation.py` is `['plastic', 'paper', 'metal', 'glass']`,
which is wrong for the weights it loads, so the Pi printed the wrong label even when the
argmax was right. I've left it as it was, with a comment on it, because it's part of what
went wrong. Use `inference.py` if you want the correct label.

The YOLO11 checkpoints on Hugging Face are stock Ultralytics releases under AGPL-3.0.
The MobileNetV3-Small weights are mine, trained from torchvision ImageNet weights.
