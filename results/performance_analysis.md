# Model Performance Analysis

## Final Test Accuracy
The final test accuracy on the 10-species iNaturalist dataset was evaluated after fine-tuning MobileNetV3. For precise values, refer to `evaluation_metrics.json`. Since the model was only trained for a few epochs (due to CPU limitations), the performance might not be optimal, and some visually similar species might be confused.

## Architectural Changes for Improvement
To reach a higher target accuracy (e.g., >80%), the following changes are recommended:
1. **Larger Backbone**: Switch from `MobileNetV3` to a more complex architecture like `ResNet50` or `EfficientNet-B4` that has a higher capacity for learning fine-grained features necessary for distinguishing visually similar species.
2. **Longer Training**: Train for more epochs (e.g., 20-50) with an learning rate scheduler (like `ReduceLROnPlateau`) to ensure full convergence.
3. **Data Augmentation**: Implement more rigorous augmentation (e.g., MixUp, CutMix, or heavier color jittering) to improve generalization.
4. **Hardware**: Leverage GPU acceleration (e.g., NVIDIA CUDA or MPS) to make training faster and enable larger batch sizes and more complex models.
5. **Class Imbalance Handling**: If any species were slightly underrepresented, use weighted loss functions or oversampling techniques.
