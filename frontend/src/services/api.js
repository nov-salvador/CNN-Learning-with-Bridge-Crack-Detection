const API_URL = "http://localhost:8000"

export const predictImage = async (file) => {
  const formData = new FormData()
  formData.append("file", file)
  const response = await fetch(`${API_URL}/predict`, {method: "post", body: formData})
  if (!response.ok) {
    const errorData = await response.json()
    throw new Error( errorData.detail || "Prediction request failed")
  }
  return response.json()
}