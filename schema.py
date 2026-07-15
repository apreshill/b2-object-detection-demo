"""Pixeltable schema for building an object-detection training set from videos.

Hierarchy:
    videos (table)            source videos + metadata
    -> frames (view)          one row per extracted frame (frame_iterator, 1 fps), YOLOX detections
    -> training_frames (view) frames with at least one detection, resized to the model input size,
                              with bounding boxes rescaled to match

Create or update the schema with:
    pxt schema update schema.py od_demo
"""

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.catalog.model import Column

TableModel = pxt.model_base()

# Model input size (square); boxes are rescaled to this canvas.
IMG_SIZE = 640

B2_BASE = "https://s3.us-west-004.backblazeb2.com/ai-campfire/od-training-demo/"
B2_FRAMES_OVERLAY_DEST = f"{B2_BASE}frame-overlays/"
B2_TRAINING_OVERLAY_DEST = f"{B2_BASE}training-overlays/"


class Videos(TableModel, name="videos"):
    """Source videos; insert local files or S3/HTTP URLs."""

    video: pxt.Required[pxt.Video]
    metadata = pxtf.video.get_metadata(video)  # noqa: F821


class Frames(TableModel, name="frames", base=Videos, iterator=pxtf.video.frame_iterator(Videos.video, fps=1.0)):
    """One row per extracted frame; the frame image is unstored and extracted on demand."""

    # Pseudo-labels: {'bboxes': [[x1, y1, x2, y2], ...], 'scores': [...], 'labels': [...]}.
    detections = pxtf.yolox.yolox(frame, model_id="yolox_s", threshold=0.5)  # noqa: F821
    num_detections = detections.bboxes.len()
    overlay = Column(
        value=pxtf.vision.bboxes_draw(frame, detections.bboxes, labels=detections.labels),  # noqa: F821
        destination=B2_FRAMES_OVERLAY_DEST,
    )


class TrainingFrames(TableModel, name="training_frames", base=Frames.where(Frames.num_detections > 0)):
    """Training samples: model-input-sized images with boxes rescaled to match."""

    image = Frames.frame.resize([IMG_SIZE, IMG_SIZE])
    boxes = pxtf.vision.bboxes_resize_canvas(
        Frames.detections.bboxes,
        "xyxy",
        canvas_width=Frames.frame.width,
        canvas_height=Frames.frame.height,
        new_canvas_width=IMG_SIZE,
        new_canvas_height=IMG_SIZE,
    )
    labels = Frames.detections.labels
    training_overlay = Column(
        value=pxtf.vision.bboxes_draw(image, boxes, labels=labels),  # noqa: F821
        destination=B2_TRAINING_OVERLAY_DEST,
    )
