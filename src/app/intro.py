import streamlit as st


def main(**kwargs):
    st.header("Video Anomaly Detection Dashboard")
    st.write("Title: Inappropriate Content Identitification in Social Media")
    st.divider()
    st.subheader("About")

    st.write(
        """
    This project aims to detect unusual events in videos using machine learning. 
    A video anomaly detection model is designed to identify anomalous frames/snippets within videos of any length. 
    We trained our models with weakly-supervised learning approach, i.e., these models, while being trained on video-level labels, are being evaluated on frame-level labels at test time.
    """
    )

    st.subheader("Our Models")
    st.write(
        """
    We have trained several models for this task on the XD-Violence dataset. You can find more information about each model, including their ROC curves and PR curves below.
    """
    )

    st.markdown(
        """
        All models are trained with 50 epochs and the checkpoint with the best validation AP is saved. 

        | Model                    | Feature  | Average Precision (AP) | ROC-AUC    | Best Epoch | Configuration                                                                                                                                  |
        | ------------------------ | -------- | ---------------------- | ---------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
        | SVM Baseline             | C3D      | 0.6146                 | 0.5        | 18         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/svm_baseline/baseline-c3d.yaml)       |
        | SultaniNet               | C3D      | 0.53                   | 0.7751     | 40         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/sultani_net/sultaninet-c3d.yaml)      |
        | PengWuNet (Offline)      | C3D      | 0.5555                 | 0.8829     | 14         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-c3d.yaml)  |
        | PengWuNet (Online)       | C3D      | 0.5561                 | 0.8747     | 14         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-c3d.yaml)  |
        | SVM Baseline             | I3D      | 0.6140                 | 0.5        | 50         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/svm_baseline/baseline-i3d.yaml)       |
        | SultaniNet               | I3D      | 0.5604                 | 0.8004     | 6          | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/sultani_net/sultaninet-i3d.yaml)      |
        | PengWuNet (Offline)      | I3D      | 0.7508                 | 0.9197     | 38         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-i3d.yaml)  |
        | PengWuNet (Online)       | I3D      | 0.7303                 | 0.9132     | 38         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-i3d.yaml)  |
        | SVM Baseline             | Swin     | 0.6140                 | 0.5        | 50         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/svm_baseline/baseline-swin.yaml)      |
        | SultaniNet               | Swin     | 0.7407                 | 0.903      | 12         | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/sultani_net/sultaninet-swin.yaml)     |
        | **PengWu-Net (Offline)** | **Swin** | **0.7918**             | **0.9304** | 4          | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-swin.yaml) |
        | PengWuNet (Online)       | Swin     | 0.742                  | 0.9156     | 4          | [link](https://github.com/hongjiaherng/inappropriate-video-detection/blob/main/anomaly-detection/configs/pengwu_net/hlnet-ctx_len_5-swin.yaml) |
        """
    )
    st.write("")

    tab1, tab2 = st.tabs(["PR Curve", "ROC Curve"])
    with tab1:
        _, pr, _ = st.columns([0.15, 0.7, 0.15])
        pr.image("https://github.com/hongjiaherng/vid-anomaly-app/blob/main/assets/pr-curves-all-models.png?raw=true")

    with tab2:
        _, roc, _ = st.columns([0.15, 0.7, 0.15])
        roc.image("https://github.com/hongjiaherng/vid-anomaly-app/blob/main/assets/roc-curves-all-models.png?raw=true")

    st.subheader("How to Use This Application")
    st.write(
        """
    There are three ways to use this application:
    1. Upload your own video. 
    2. Paste a YouTube link.
    3. Open your webcam. 

    The application will then run the selected model on the video and display the results.
    """
    )

    st.subheader("Contact Us")
    st.write(
        """
    If you have any questions or feedback, please feel free to contact us at [jiaherng2001@gmail.com](mailto:jiaherng2001@gmail.com)
    """
    )
