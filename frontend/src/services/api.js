const API_URL = import.meta.env.VITE_API_URL

const compressImage = (file, maxDimension = 1024, quality = 0.8) => {
  return new Promise((resolve, reject) => {
    const image = new Image()
    const objectUrl = URL.createObjectURL(file)

    image.onload = () => {
      if (image.width <= maxDimension && image.height <= maxDimension) {
        URL.revokeObjectURL(objectUrl)
        resolve(file)
        return
      }
      let { width, height } = image

      if (width > height) {
        height = Math.round((height / width) * maxDimension)
        width = maxDimension
      } else {
        width = Math.round((width / height) * maxDimension)
        height = maxDimension
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
  if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) {
   throw new Error("Only JPEG, JPG and PNG images are supported.")
  }
  console.log("Original:", file.size)
  const compressedImage = await compressImage(file)
  console.log("Compressed:", compressedImage.size)
  const formData = new FormData()
  formData.append("file", compressedImage, "compressed-image.jpg")
  const response = await fetch(`${API_URL}/predict`, {method: "post", body: formData})
  if (!response.ok) {
    const errorData = await response.json()
    throw new Error( errorData.detail || "Prediction request failed")
  }
  return response.json()
}