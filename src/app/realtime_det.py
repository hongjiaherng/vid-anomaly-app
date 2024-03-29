import concurrent.futures
import logging
import queue
import threading
import time
from typing import Union

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from app.utils.common import configure_settings_sidebar, init_device
from app.utils.model_builder import load_backbone, load_detector, predict_pipeline
from app.utils.turn import get_ice_servers
from streamlit_webrtc import RTCConfiguration, VideoHTMLAttributes, VideoProcessorBase, webrtc_streamer

logger = logging.getLogger(__name__)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
FRAMES_TO_UPDATE_AT_ONCE = 8


class BatchedFramesProcessor(VideoProcessorBase):
    WIDTH = CAMERA_WIDTH
    HEIGHT = CAMERA_HEIGHT

    def __init__(self) -> None:
        self._sampling_rate = None
        self._clip_len = None
        self._backbone = None
        self._detector = None
        self._clip_preprocessor = None
        self._device = None

        self._frames_buffer = []
        self._frames_buffer_lock = threading.Lock()
        self._score_queue = queue.Queue()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Batch frames sufficiently to become a clip
        with self._frames_buffer_lock:
            self._frames_buffer.append(cv2.resize(frame.to_ndarray(format="rgb24"), (self.WIDTH, self.HEIGHT)))

            if len(self._frames_buffer) == self._clip_len * self._sampling_rate:
                # Process the clip
                logger.info(f"Done batching frames into a clip of length {len(self._frames_buffer)}")
                self._thread_pool.submit(self._process_clip)

        return frame

    def set_attrs_on_render(self, **kwargs):
        self._sampling_rate = kwargs["sampling_rate"]
        self._clip_len = kwargs["clip_len"]
        self._backbone = kwargs["backbone"]
        self._detector = kwargs["detector"]
        self._clip_preprocessor = kwargs["clip_preprocessor"]
        self._device = kwargs["device"]

    def _process_clip(self):
        with self._frames_buffer_lock:
            current = self._frames_buffer.copy()
            self._frames_buffer.clear()

        current = np.stack(current, axis=0)  # (sampling_rate * clip_len, H, W, C)
        logger.info(f"Processing clip {current.shape=}")
        self._make_prediction(current)

    def _make_prediction(self, clip: np.ndarray):  # (sampling_rate * clip_len, H, W, C)
        # Sample frames based on sampling rate
        clip = torch.from_numpy(clip[:: self._sampling_rate].transpose(0, 3, 1, 2)).unsqueeze(0)  # (1, clip_len, C, H, W)
        clip = self._clip_preprocessor({"inputs": clip, "meta": {}})  # {"inputs": (1, crop, C, clip_len, H, W)}
        score = predict_pipeline(clip_dict=clip, backbone=self._backbone, detector=self._detector, device=self._device)

        # Enqueue the score
        self._score_queue.put(score)
        logger.info(f"Enqueued score {score=}")

    def get_latest_score(self) -> Union[float, None]:
        try:
            return self._score_queue.get_nowait()
        except queue.Empty:
            return None


def main(**kwargs):
    st.header("Video Anomaly Detection Dashboard")
    st.markdown("Run real-time video anomaly detection on your webcam.")
    st.divider()

    model_name, feature_name, ckpt_type, threshold, enable_gpu = configure_settings_sidebar()
    device = init_device(enable_gpu)
    backbone_model, _, clip_preprocessor, sampling_strategy = load_backbone(feature_name, device, crop_type="center")
    detector_model = load_detector(model_name, feature_name, ckpt_type, device)

    webrtc_ctx = webrtc_streamer(
        key="webcam",
        video_transformer_factory=BatchedFramesProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, style={"width": "100%"}, muted=True),
        media_stream_constraints={
            "video": {
                "width": {"min": CAMERA_WIDTH, "ideal": CAMERA_WIDTH, "max": CAMERA_WIDTH},
                "height": {"min": CAMERA_HEIGHT, "ideal": CAMERA_HEIGHT, "max": CAMERA_HEIGHT},
            },
        },
        async_processing=True,
    )

    if webrtc_ctx.video_processor:  # IMPORTANT: set_attrs_on_render() must be called after webrtc_ctx.video_processor is initialized
        webrtc_ctx.video_processor.set_attrs_on_render(
            sampling_rate=sampling_strategy["sampling_rate"],
            clip_len=sampling_strategy["clip_len"],
            backbone=backbone_model,
            detector=detector_model,
            clip_preprocessor=clip_preprocessor,
            device=device,
        )

    st.divider()

    if webrtc_ctx.state.playing:
        frame_count = 0

        st.markdown("#### Result")
        st.write("\n")

        while True:
            if webrtc_ctx.video_processor is None:  # if stopped it will be None
                break
            score = webrtc_ctx.video_processor.get_latest_score()
            if score is not None:
                # score = round(score, 4)  # rounding off for display purpose
                logger.info(f"Score: {score=}")

                for frame_start in range(
                    frame_count,
                    frame_count + sampling_strategy["sampling_rate"] * sampling_strategy["clip_len"],
                    FRAMES_TO_UPDATE_AT_ONCE,
                ):
                    # Update 8 frames at a time
                    if frame_start == 0:
                        chart_placeholder = st.line_chart(
                            data=pd.DataFrame(
                                {
                                    "frame": np.array([frame_start], dtype=int),
                                    "anomaly score": np.array([score], dtype=np.float32),
                                    "threshold": np.array([threshold], dtype=np.float32),
                                }
                            ),
                            x="frame",
                            y=["anomaly score", "threshold"],
                        )
                    else:
                        chart_placeholder.add_rows(
                            pd.DataFrame(
                                {
                                    "frame": np.array([frame_start], dtype=int),
                                    "anomaly score": np.array([score], dtype=np.float32),
                                    "threshold": np.array([threshold], dtype=np.float32),
                                }
                            )
                        )
                    frame_count += FRAMES_TO_UPDATE_AT_ONCE
                    time.sleep(0.3)  # Sleep to make the chart smoother
            else:
                time.sleep(0.3)  # Sleep to avoid busy waiting
