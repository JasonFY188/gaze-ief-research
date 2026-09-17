"""
Drop-in replacement for Pipeline that uses the IEF refinement model.

Swap usage:

    # Before (plain L2CS-Net)
    from l2cs import Pipeline
    pipe = Pipeline(weights="L2CSNet_gaze360.pkl", arch="ResNet50", device="cuda")

    # After (IEF on Gaze360 backbone)
    from l2cs import IEFPipeline
    pipe = IEFPipeline(
        backbone_weights="L2CSNet_gaze360.pkl",
        head_weights="checkpoints/ief_gaze360/best_fold0.pt",
        device="cuda",
    )

    # After (IEF on MPIIGaze backbone, fold N)
    from l2cs import IEFPipeline
    pipe = IEFPipeline(
        backbone_weights="models/MPIIGaze/fold0.pkl",
        head_weights="checkpoints/ief_mpiigaze/best_fold0.pt",
        device="cuda",
    )

The .step() and .predict_gaze() methods return the same types as Pipeline.
The RefinementConfig (bin count, bin width, angle offset) is read from the
checkpoint — you do not need to specify it manually.
"""

import math
import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from .utils import prep_input_numpy
from .results import GazeResultContainer
from .model import L2CS
from .wrapper import L2CSWrapper
from .refinement_head import RefinementConfig, IEFGazeModel


class IEFPipeline:
    """IEF gaze pipeline — same interface as Pipeline, IEF model under the hood.

    Args:
        backbone_weights: path to the frozen backbone .pkl
            (Gaze360: L2CSNet_gaze360.pkl  or  MPIIGaze: fold{N}.pkl)
        head_weights: path to the IEF head checkpoint .pt saved by
            train_refinement.py or train_ief_mpiigaze.py
        device: "cpu", "cuda", or "cuda:0" etc.
        include_detector: run RetinaFace face detection before gaze prediction
        confidence_threshold: minimum face-detection score
    """

    def __init__(
        self,
        backbone_weights: Union[str, pathlib.Path],
        head_weights: Union[str, pathlib.Path],
        device: str = "cpu",
        include_detector: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.include_detector = include_detector
        self.confidence_threshold = confidence_threshold

        # Load head checkpoint — it also carries the full RefinementConfig
        ckpt = torch.load(head_weights, map_location="cpu", weights_only=False)
        cfg: RefinementConfig = ckpt["cfg"]
        self.cfg = cfg

        # Build and load backbone
        base = L2CS(
            torchvision.models.resnet.Bottleneck,
            [3, 4, 6, 3],
            num_bins=cfg.num_bins,
        )
        state = torch.load(backbone_weights, map_location="cpu", weights_only=False)

        # MPIIGaze fold*.pkl files were saved under DataParallel (module. prefix)
        if any(k.startswith("module.") for k in state.keys()):
            dp = nn.DataParallel(base)
            dp.load_state_dict(state)
            base = dp.module
        else:
            base.load_state_dict(state)

        wrapper = L2CSWrapper(base)

        # Assemble IEF model and load trained head weights
        self.model = IEFGazeModel(wrapper, cfg).to(self.device)
        self.model.head.load_state_dict(ckpt["head_state_dict"])
        self.model.eval()

        if include_detector:
            from face_detection import RetinaFace
            if self.device.type == "cpu":
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=self.device.index)

    def step(self, frame: np.ndarray) -> GazeResultContainer:
        face_imgs = []
        bboxes    = []
        landmarks = []
        scores    = []

        if self.include_detector:
            faces = self.detector(frame)
            if faces is not None:
                for box, landmark, score in faces:
                    if score < self.confidence_threshold:
                        continue
                    x_min = max(int(box[0]), 0)
                    y_min = max(int(box[1]), 0)
                    x_max = int(box[2])
                    y_max = int(box[3])
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                if face_imgs:
                    pitch, yaw = self.predict_gaze(np.stack(face_imgs))
                else:
                    pitch = np.empty((0, 1))
                    yaw   = np.empty((0, 1))
            else:
                pitch = np.empty((0, 1))
                yaw   = np.empty((0, 1))
        else:
            pitch, yaw = self.predict_gaze(frame)
            bboxes = landmarks = scores = []

        return GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes) if bboxes else np.empty((0, 4)),
            landmarks=np.stack(landmarks) if landmarks else np.empty((0, 5, 2)),
            scores=np.stack(scores) if scores else np.empty((0,)),
        )

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        """Run IEF inference and return (pitch_rad, yaw_rad) arrays.

        Returns the final refined step's prediction in radians, matching the
        output format of Pipeline.predict_gaze().
        """
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame.to(self.device)
        else:
            raise RuntimeError(f"Unsupported input type: {type(frame)}")

        with torch.no_grad():
            steps = self.model(img)

        # Use the final refined step (steps[-1]), not the raw backbone (steps[0])
        pitch_logits, yaw_logits = steps[-1]

        cfg = self.cfg
        idx        = torch.arange(cfg.num_bins, dtype=torch.float32, device=self.device)
        bin_angles = idx * cfg.bin_width_deg - cfg.angle_offset_deg   # degrees

        pitch_deg = (torch.softmax(pitch_logits, dim=1) * bin_angles).sum(dim=1)
        yaw_deg   = (torch.softmax(yaw_logits,   dim=1) * bin_angles).sum(dim=1)

        pitch_rad = pitch_deg.cpu().numpy() * math.pi / 180.0
        yaw_rad   = yaw_deg.cpu().numpy()   * math.pi / 180.0

        return pitch_rad, yaw_rad
