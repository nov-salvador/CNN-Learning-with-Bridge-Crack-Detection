const API_URL = import.meta.env.VITE_API_URL

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