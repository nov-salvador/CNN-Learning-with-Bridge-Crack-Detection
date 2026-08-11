const API_URL = import.meta.env.VITE_API_URL

const compressImage = (file, maxDimension = 1024, quality = 0.8) => {
  return new Promise((resolve, reject) => {
    const image = new Image()
    const objectUrl = URL.createObjectURL(file)

    image.onload = () => {
      let { width, height } = image

      // Keep the original aspect ratio
      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          height = Math.round((height / width) * maxDimension)
          width = maxDimension
        } else {
          width = Math.round((width / height) * maxDimension)
          height = maxDimension
        }
      }

      const canvas = document.createElement("canvas")
      canvas.width = width
      canvas.height = height

      const context = canvas.getContext("2d")
      context.drawImage(image, 0, 0, width, height)

      canvas.toBlob(
        (blob) => {
          URL.revokeObjectURL(objectUrl)

          if (!blob) {
            reject(new Error("Failed to compress image"))
            return
          }

          resolve(blob)
        },
        "image/jpeg",
        quality
      )
    }

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl)
      reject(new Error("Failed to load image"))
    }

    image.src = objectUrl
  })
}
export const predictImage = async (file) => {
  const compressedImage = await compressImage(file)
  const formData = new FormData()
  formData.append("file", compressedImage, "compressed-image.jpg")
  const response = await fetch(`${API_URL}/predict`, {method: "post", body: formData})
  if (!response.ok) {
    const errorData = await response.json()
    throw new Error( errorData.detail || "Prediction request failed")
  }
  return response.json()
}