package types

// FrameTask represents a single frame sent to a worker for processing
type FrameTask struct {
	Index int
	Data  []byte
}

// FaceResult matches the JSON structure coming back from File 1 (Python)
type FaceResult struct {
	Loc     []int     `json:"loc"`     // [top, right, bottom, left]
	Vec     []float64 `json:"vec"`     // 512-d face encoding
	Thumb   []byte    `json:"thumb"`   // Raw JPEG bytes
	Quality float64   `json:"quality"` // Detection score * Area
}

// ErrorResult captures the error object returned by Python on failure
type ErrorResult struct {
	Error string `json:"error"`
}
