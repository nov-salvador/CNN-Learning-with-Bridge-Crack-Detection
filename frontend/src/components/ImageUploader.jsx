function ImageUploader({preview, selectedImage, onImageChange}) {
  return (
    <div className="upload-box">
      <p>Upload an image</p>

      <input
        type="file"
        accept="image/*"
        onChange={onImageChange}
      />

      {preview && (
        <div>
          <img
            src={preview}
            alt="Selected bridge"
            className="image-preview"
          />

          <p>{selectedImage.name}</p>
        </div>
      )}
    </div>
  )
}

export default ImageUploader