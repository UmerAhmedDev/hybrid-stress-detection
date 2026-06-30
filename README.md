4. The final score (0–100) is mapped to a stress level:

| Score Range | Stress Level |
|-------------|--------------|
| 80 – 100 | Critical Stress |
| 60 – 79 | High Stress |
| 40 – 59 | Moderate Stress |
| Below 40 | Low Stress |

## Tech Stack

- **Python 3** – Core programming language
- **OpenCV (cv2)** – Webcam access and face detection (Haar Cascade)
- **PyTorch** – Deep learning framework for emotion recognition
- **Torchvision** – Pretrained ResNet50 architecture
- **NumPy** – Numerical computations
- **Pandas** – Dataset loading and preprocessing
- **Threading** – Parallel execution of camera and survey modules
- **ResNet50 CNN** – Facial emotion classification model

## Getting Started

### Prerequisites

- Python 3.x
- A working webcam

### Installation

```bash
pip install opencv-python torch torchvision numpy pandas
```

### Required Files

Make sure the following files are present in the project directory before running:

- `best_rafdb_resnet50.pth` — trained ResNet50 emotion recognition model
- `Stress Indicators Dataset for Mental Health Classification.csv` — survey dataset

> **Note:** These files are not included in this repository due to size constraints. See [Dataset Sources](#dataset-sources) below to download them.

### Running the Project

```bash
python main.py
```

1. The webcam window will open and begin capturing facial emotions.
2. Survey questions will be asked in the terminal — answer them as prompted.
3. Press **`q`** to close the camera window once the survey is complete.
4. The program will output:
   - Camera-based stress score
   - Survey-based stress score
   - Final combined stress score and stress level

## Dataset Sources

- **Facial Emotion Dataset (RAF-DB):** [Kaggle](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
- **Survey Dataset (Stress Indicators for Mental Health Classification):** [Mendeley Data](https://data.mendeley.com/datasets/2gsjv8m7ch/1)

## Output Interpretation

- Stress scores range from **0 to 100**
- Final score = `25% Camera Stress + 75% Survey Stress`
- See the [Stress Levels table](#how-it-works) above for interpretation

## Disclaimer

This tool is intended for educational and research purposes only. It is **not** a clinical or diagnostic tool and should not be used as a substitute for professional mental health advice.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
