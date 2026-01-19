package types

// FrameTask represents a single frame sent to a worker for processing
type FrameTask struct {
	Index int
	Data  []byte
}

// FaceResult matches the JSON structure coming back from File 1 (Python)
type FaceResult struct {
	Loc      []int     `json:"loc"`       // [top, right, bottom, left]
	Vec      []float64 `json:"vec"`       // 512-d face encoding
	ThumbB64 string    `json:"thumb_b64"` // Base64 encoded JPEG thumbnail
	Quality  float64   `json:"quality"`   // Detection score * Area
}

// ErrorResult captures the error object returned by Python on failure
type ErrorResult struct {
	Error string `json:"error"`
}
