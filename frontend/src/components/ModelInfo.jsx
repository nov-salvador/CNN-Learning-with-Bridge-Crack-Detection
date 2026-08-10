
function ModelInfo() {
  return (
    <section className="model-info">
      <h2>About the Model</h2>

      <p>
        This application uses a convolutional neural network
        trained to detect cracks in bridge images.
      </p>

      <div className="model-details">
        <div>
          <span>Architecture:</span>
          <strong>{` ResNet18`}</strong>
        </div>

        <div>
          <span>Approach:</span>
          <strong>{` Transfer Learning`}</strong>
        </div>

        <div>
          <span>Framework:</span>
          <strong>{` PyTorch`}</strong>
        </div>

        <div>
          <span>Input Size:</span>
          <strong>{` 224 × 224`}</strong>
        </div>
      </div>

      <p className="model-note">
        The model was fine-tuned on a bridge crack dataset
        containing crack and no-crack images.
      </p>
    </section>
  )
}

export default ModelInfo