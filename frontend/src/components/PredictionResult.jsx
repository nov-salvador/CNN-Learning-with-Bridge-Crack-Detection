
function PredictionResult({prediction}) {
  if (!prediction){
    return null
  }

  const isCrack = prediction.prediction === "crack"
  return (
    <div className={`result ${isCrack ? "crack" : "no-crack"}`}>
      <h2>Prediction Result</h2>

      <h3>
        {isCrack ? "CRACK DETECTED" : "NO CRACK DETECTED"}
      </h3>

      <p className="confidence">
        {(prediction.confidence).toFixed(2)}%
      </p>
      <p>
        {isCrack
          ? "The model detected signs of a crack in this image."
          : "The model did not detect a crack in this image."}
      </p>
    </div>
  )
}

export default PredictionResult