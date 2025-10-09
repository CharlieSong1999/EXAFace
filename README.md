# Extreme amodal face detection

Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches.



# Train

# Inference

# Demo

