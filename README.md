# Img2Line AI Service

## This project is trained with customed dataset: flatart borderless

# TEED: Tiny and Efficient Edge Detection

TEED is a research project for fast, lightweight, and accurate edge detection in images, based on a custom neural network architecture. It includes tools for data preprocessing, model training, evaluation, and post-processing, supporting both standard and custom datasets. The project also provides utilities for converting between raster (PNG) and vector (SVG) formats, leveraging the [vtracer](https://pypi.org/project/vtracer/) library and [rsvg-convert](https://wiki.gnome.org/Projects/LibRsvg).

## Features

- **High-Performance Edge Detection**: Fast inference with precision, recall, and F1 metrics logged to TensorBoard
- **Flexible Training**: Train from scratch or fine-tune from pretrained weights; configurable loss functions and learning schedules
- **Data Preprocessing**: Convert input PNGs → SVGs → resize to 1024×1024 → convert back to RGB PNGs
- **Post-Processing**: Vectorize PNG results to SVGs; rasterize SVGs to PNGs for downstream applications
- **Evaluation Suite**: Compute precision, recall, and F1 scores with visual outputs for validation

## Installation

### Python Dependencies (Python ≥ 3.7)

```bash
pip install -r requirements.txt
pip install vtracer
```

### System Packages

**macOS:**
```bash
brew install librsvg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install librsvg2-bin
```

**Windows:** See [librsvg Windows builds](https://wiki.gnome.org/Projects/LibRsvg)

## Project Structure

```
├── checkpoints/         # Pretrained & custom model weights
├── data/                # Preprocessed images for training & testing
├── input_data/          # Raw PNGs for processing
├── output_data_png/     # Post-processed PNG outputs
├── output_data_svg/     # Post-processed SVG outputs
├── result/              # Inference results
├── ted.py               # TEED model definition
├── loss2.py             # Training loss functions
├── dataset.py           # Dataset loader and transforms
├── preprocessor.py      # PNG↔SVG conversion and resizing pipeline
├── main.py              # CLI for preprocessing, training, testing, and post-processing
└── utils/               # Helper modules (image I/O, activation functions, metrics)
```

## Quick Start

1. Add your raw PNGs to `input_data/`

2. Run the end-to-end pipeline:
   ```bash
   python main.py --choose_test_data -1
   ```

3. Retrieve outputs:
   - **Edge maps**: `result/`
   - **SVG vectors**: `output_data_svg/`
   - **Rasterized PNGs**: `output_data_png/`

## Evaluation

Launch TensorBoard to view training and validation metrics:

```bash
tensorboard --logdir runs/
```

**Metrics logged:** precision, recall, and F1 score per epoch

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [vtracer](https://pypi.org/project/vtracer/) for raster-to-vector conversion
- [librsvg](https://wiki.gnome.org/Projects/LibRsvg) for SVG rasterization
- Research by Xavier Soria Poma and contributors

For questions or contributions, please open an issue or submit a pull request.

