# Build an object detection training set from videos

This webinar demo turns raw videos into a PyTorch object detection training set with a declarative Pixeltable schema. The pipeline has three levels:

```text
videos (table)             source videos and metadata
  -> frames (view)         one frame per second with YOLOX detections as pseudo labels
  -> training_frames       frames with detections, resized to 640 by 640 pixels
     (view)                with bounding boxes rescaled to match
```

Everything downstream of `videos` is computed automatically and incrementally. Insert a video and Pixeltable produces the training samples. Insert another video and Pixeltable processes only its new frames.

The demo uses public videos from the [Multimedia Commons](http://mmcommons.org/) S3 bucket. It shows how to:

* Extract video frames at one frame per second.
* Use YOLOX detections as pseudo labels.
* Draw detection overlays and store them in Backblaze B2.
* Resize each image and its bounding boxes together.
* Export the result as a Pixeltable PyTorch dataset.
* Apply random image and bounding box transforms in a PyTorch DataLoader.

## Run the demo

Install [uv](https://docs.astral.sh/uv/), then run:

```shell
uv sync
uv run jupyter lab
```

Open `object-detection-demo.ipynb` and run the cells in order.

You can also create or update the schema from the terminal:

```shell
uv run pxt schema update schema.py od_demo
```

## How the training data is prepared

Object detection training uses two kinds of transforms.

Pixeltable handles deterministic work such as decoding, resizing, and rescaling boxes. These results are computed once, cached, and shared.

The PyTorch DataLoader handles random transforms such as flipping and cropping. They run again during each training epoch. Geometric transforms use `torchvision.transforms.v2` and `tv_tensors.BoundingBoxes`, so each transform changes the image and its boxes together.

`to_pytorch_dataset()` writes the query result to a local cache and returns an `IterableDataset`. Later epochs read from that cache instead of running the Pixeltable query again.

## Backblaze B2

The schema stores detection overlays in the `ai-campfire` B2 bucket. Set its write credentials before you run the insert cells:

```shell
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

You can change the two destination constants near the top of `schema.py` to use another location.

This demo uses the same Pixeltable and B2 setup shown in [Backblaze B2 and Pixeltable Examples](https://github.com/backblaze-b2-samples/b2-pixeltable-multimodal-data). That project provides a fuller guide to configuring a B2 bucket, checking access, extracting video frames, and storing computed media back in B2.

The Pixeltable guide to [uploading media to S3 and other cloud storage](https://docs.pixeltable.com/howto/cookbooks/data/data-export-s3) explains per column destinations, global destination settings, supported storage providers, file URLs, and signed URLs.

## Schema update

This demo uses Pixeltable class-based schemas with `pxt schema update`. That command reads the schema from `schema.py`, then creates or updates the matching tables and views.

The uv environment installs Pixeltable from PyPI.
