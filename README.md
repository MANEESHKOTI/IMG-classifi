# iNaturalist Species Image Classifier

This project builds an end-to-end image classification pipeline using the iNaturalist dataset. It acquires data from a public API, implements transfer learning by fine-tuning a MobileNetV3 model with PyTorch, and interprets model predictions using Grad-CAM.

## Core Technologies
- **iNaturalist API**: Data source for wildlife observations.
- **Transfer Learning & MobileNetV3**: Efficient convolutional neural network adapted for species classification.
- **PyTorch**: Deep learning framework used for model training and evaluation.
- **Grad-CAM**: Explainable AI technique to visualize model decisions.
- **Docker**: Containerization for reproducible environments.

## Project Structure
```text
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
│   └── grad_cam_outputs/
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_preprocess_data.py
│   ├── 03_train_model.py
│   ├── 04_evaluate_model.py
│   └── 05_run_grad_cam.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Setup and Execution

1. Clone the repository and navigate to the root directory.
2. Build the Docker environment:
   ```bash
   docker-compose up --build -d
   ```
3. Run an interactive shell in the container:
   ```bash
   docker-compose run app bash
   ```
4. Execute the pipeline scripts in order:
   ```bash
   python scripts/01_fetch_data.py
   python scripts/02_preprocess_data.py
   python scripts/03_train_model.py
   python scripts/04_evaluate_model.py
   python scripts/05_run_grad_cam.py
   ```
