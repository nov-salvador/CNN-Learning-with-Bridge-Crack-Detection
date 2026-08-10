import { useState } from "react"
import {predictImage} from "./services/api"
import ImageUploader from "./components/ImageUploader"
import PredictionResult from "./components/PredictionResult"
import ModelInfo from "./components/ModelInfo"

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0]
    if (!file){
      return
    }
    setSelectedImage(file)
    setPreview(URL.createObjectURL(file))
  }

  const handlePredict = async() => {
    if (!selectedImage) {
      setError("Please select an image first")
      return
    }
    setLoading(true)
    setError(null)
    setPrediction(null)

    
    try {
      const data = await predictImage(selectedImage)
      setPrediction(data)
    } catch(error){
      setError(error.message)
    } finally {
      setLoading(false)
    }

  }

  return (
    <>
      <div className='app'>
        <header>
          <h1>Bridge Crack Detection</h1>
          <p>Upload a bridge or concrete image to detect potential crack</p>
        </header>

        <main>
          <ImageUploader preview={preview} selectedImage={selectedImage} onImageChange={handleImageChange} />
          <button onClick={handlePredict} disabled={loading}> {loading ? "Predicting..." : "Predict"}</button>
          {loading && <p>Predicting...</p>}
          {error && <p className="error-message">{error}</p>}
          <PredictionResult prediction={prediction} />
          <ModelInfo />
        </main>
      </div>
    </>
  )
}

export default App
