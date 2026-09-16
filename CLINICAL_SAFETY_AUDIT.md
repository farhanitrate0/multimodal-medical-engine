# Clinical Safety and Audit Documentation

## System Intended Use and Architecture
The Multimodal Telemetry Engine is an experimental clinical decision support tool. It is not built for primary diagnosis. 

* **Radiological Diagnosis Mode:** This uses a finetuned ResNet50 network to check chest radiographs for signs of pneumonia consolidation.
* **Spatial Telemetry Mode:** This uses spatial tracking to monitor patient movements like face mesh and hand positions. This data is intended for virtual reality environments.
* **Scope Restriction:** All outputs are based on probabilities and require human in the loop oversight. They absolutely do not replace a formal review by a radiologist.

## Explainability and Clever Hans Mitigation
Medical artificial intelligence needs strict transparency. We have to make sure the model looks at real anatomical pathology and does not just cheat by finding background artifacts.

* **Gradient Mapping Integration:** The engine uses Gradient weighted Class Activation Mapping on the final convolutional block. This visually highlights the exact pixels that caused the model to output its confidence score.
* **Artifact Validation:** We actively check the heatmaps to confirm the model focuses on the lung fields. It should not react to image borders, phone screen bezels, or hospital watermarks.
* **Sanity Checks:** The code includes support for occlusion testing and weight randomization. These checks mathematically prove that our heatmaps accurately reflect the actual decision boundaries of the model.

## Data Privacy and Memory Architecture
We protect patient data privacy at the foundational level to meet standard data handling principles.

* **In Memory Processing:** Telemetry frames and image scans stream through WebSockets as base64 payloads and decode directly into RAM.
* **No Persistent Storage:** The system strictly forbids saving temporary diagnostic frames or spatial coordinates to a physical disk. This eliminates the risk of residual data leaks.

## Known Failure Modes and Limitations
* **Out of Distribution Degradation:** The diagnostic mode was trained exclusively on chest radiographs. If you feed it random pictures, it will produce erratic and unstable heatmaps.
* **Optical Artifacts:** Pointing a webcam at a physical monitor creates glare and moire patterns. These lighting issues can artificially shift the neural network focus.
* **Kinematic Occlusion:** Bad lighting or physical objects blocking the camera will cause the spatial mesh to temporarily lose tracking.
