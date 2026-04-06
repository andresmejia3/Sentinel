package cmd

import (
	"context"
	"image"
	"image/color"
	"os"
	"strings"
	"testing"

	"github.com/andresmejia3/sentinel/internal/store"
	"github.com/andresmejia3/sentinel/internal/types"
	"github.com/spf13/cobra"
)

func TestCommandNeedsDB(t *testing.T) {
	redact := &cobra.Command{Use: "redact"}
	redact.Flags().String("mode", "blur-all", "")
	redact.Flags().String("target", "", "")

	if err := redact.Flags().Set("mode", "blur-all"); err != nil {
		t.Fatalf("failed to set mode: %v", err)
	}
	if commandNeedsDB(redact) {
		t.Fatalf("blur-all redact command should not require DB")
	}

	if err := redact.Flags().Set("mode", "targeted"); err != nil {
		t.Fatalf("failed to set mode: %v", err)
	}
	if !commandNeedsDB(redact) {
		t.Fatalf("targeted redact command should require DB")
	}

	redactImplicit := &cobra.Command{Use: "redact"}
	redactImplicit.Flags().String("mode", "blur-all", "")
	redactImplicit.Flags().String("target", "", "")
	if err := redactImplicit.Flags().Set("target", "3"); err != nil {
		t.Fatalf("failed to set target: %v", err)
	}
	if !commandNeedsDB(redactImplicit) {
		t.Fatalf("--target should imply targeted mode and require DB")
	}

	other := &cobra.Command{Use: "scan"}
	if !commandNeedsDB(other) {
		t.Fatalf("non-redact commands should require DB")
	}
}

func TestLoadRedactTargetVariants(t *testing.T) {
	originalDB := DB
	DB = nil
	defer func() {
		DB = originalDB
	}()

	variants, err := loadRedactTargetVariants(context.Background(), "blur-all", []int{1})
	if err != nil {
		t.Fatalf("blur-all unexpectedly required DB: %v", err)
	}
	if len(variants) != 0 {
		t.Fatalf("blur-all returned variants: %+v", variants)
	}

	_, err = loadRedactTargetVariants(context.Background(), "targeted", []int{1})
	if err == nil {
		t.Fatalf("targeted mode should require DB")
	}
	if !strings.Contains(err.Error(), "requires database connection") {
		t.Fatalf("unexpected targeted error: %v", err)
	}
}

func TestValidateRedactFlagsAllowsImplicitTargetedMode(t *testing.T) {
	input, err := os.CreateTemp(t.TempDir(), "redact-input-*.mp4")
	if err != nil {
		t.Fatalf("failed to create temp input: %v", err)
	}
	input.Close()

	originalTargets := redactTargets
	defer func() {
		redactTargets = originalTargets
	}()

	redactTargets = "3"
	cmd := &cobra.Command{Use: "redact"}
	cmd.Flags().String("mode", "blur-all", "")
	cmd.Flags().String("target", "", "")
	if err := cmd.Flags().Set("target", "3"); err != nil {
		t.Fatalf("failed to set target: %v", err)
	}

	opts := &Options{InputPath: input.Name(), MatchThreshold: 0.6, DetectionThreshold: 0.5, NumEngines: 1, WorkerTimeout: "30s"}
	if err := validateRedactFlags(cmd, opts); err != nil {
		t.Fatalf("expected implicit targeted mode to validate, got: %v", err)
	}
}

func TestValidateRedactFlagsRejectsExplicitBlurAllWithTarget(t *testing.T) {
	input, err := os.CreateTemp(t.TempDir(), "redact-input-*.mp4")
	if err != nil {
		t.Fatalf("failed to create temp input: %v", err)
	}
	input.Close()

	originalTargets := redactTargets
	defer func() {
		redactTargets = originalTargets
	}()

	redactTargets = "3"
	cmd := &cobra.Command{Use: "redact"}
	cmd.Flags().String("mode", "blur-all", "")
	cmd.Flags().String("target", "", "")
	if err := cmd.Flags().Set("mode", "blur-all"); err != nil {
		t.Fatalf("failed to set mode: %v", err)
	}
	if err := cmd.Flags().Set("target", "3"); err != nil {
		t.Fatalf("failed to set target: %v", err)
	}

	opts := &Options{InputPath: input.Name(), MatchThreshold: 0.6, DetectionThreshold: 0.5, NumEngines: 1, WorkerTimeout: "30s"}
	err = validateRedactFlags(cmd, opts)
	if err == nil {
		t.Fatalf("expected validation error when --target is used with explicit blur-all")
	}
	if !strings.Contains(err.Error(), "--target cannot be used with explicit --mode blur-all") {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestValidateRedactFlagsRejectsExplicitZeroBoxScale(t *testing.T) {
	input, err := os.CreateTemp(t.TempDir(), "redact-input-*.mp4")
	if err != nil {
		t.Fatalf("failed to create temp input: %v", err)
	}
	input.Close()

	cmd := &cobra.Command{Use: "redact"}
	cmd.Flags().String("mode", "blur-all", "")
	cmd.Flags().String("target", "", "")
	cmd.Flags().Float64("box-scale", defaultRedactionBoxScale, "")
	if err := cmd.Flags().Set("box-scale", "0"); err != nil {
		t.Fatalf("failed to set box-scale: %v", err)
	}

	opts := &Options{InputPath: input.Name(), MatchThreshold: 0.6, DetectionThreshold: 0.5, NumEngines: 1, WorkerTimeout: "30s", BoxScale: 0}
	err = validateRedactFlags(cmd, opts)
	if err == nil {
		t.Fatalf("expected validation error for explicit box-scale 0")
	}
	if !strings.Contains(err.Error(), "invalid box-scale") {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestRedactFaceBlackClipsToBounds(t *testing.T) {
	img := newFilledImage(4, 4, color.RGBA{R: 10, G: 20, B: 30, A: 255})

	redactFace(img, image.Rect(-2, -2, 2, 2), "black", 1)

	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			got := rgbaAt(img, x, y)
			if x < 2 && y < 2 {
				if got != (color.RGBA{A: 255}) {
					t.Fatalf("pixel (%d,%d) = %+v, want black", x, y, got)
				}
				continue
			}
			want := color.RGBA{R: 10, G: 20, B: 30, A: 255}
			if got != want {
				t.Fatalf("pixel (%d,%d) = %+v, want %+v", x, y, got, want)
			}
		}
	}
}

func TestScaledRedactionRectExpandsAndClipsToBounds(t *testing.T) {
	rect := image.Rect(1, 1, 3, 3)
	got := scaledRedactionRect(rect, 2.0, image.Rect(0, 0, 4, 4))
	want := image.Rect(0, 0, 4, 4)
	if got != want {
		t.Fatalf("scaled rect = %v, want %v", got, want)
	}
}

func TestRedactFacePixelatesBlocks(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	setRGBA(img, 0, 0, color.RGBA{R: 10, A: 255})
	setRGBA(img, 1, 0, color.RGBA{R: 20, A: 255})
	setRGBA(img, 2, 0, color.RGBA{R: 30, A: 255})
	setRGBA(img, 3, 0, color.RGBA{R: 40, A: 255})
	setRGBA(img, 0, 1, color.RGBA{R: 50, A: 255})
	setRGBA(img, 1, 1, color.RGBA{R: 60, A: 255})
	setRGBA(img, 2, 1, color.RGBA{R: 70, A: 255})
	setRGBA(img, 3, 1, color.RGBA{R: 80, A: 255})
	setRGBA(img, 0, 2, color.RGBA{R: 90, A: 255})
	setRGBA(img, 1, 2, color.RGBA{R: 100, A: 255})
	setRGBA(img, 2, 2, color.RGBA{R: 110, A: 255})
	setRGBA(img, 3, 2, color.RGBA{R: 120, A: 255})
	setRGBA(img, 0, 3, color.RGBA{R: 130, A: 255})
	setRGBA(img, 1, 3, color.RGBA{R: 140, A: 255})
	setRGBA(img, 2, 3, color.RGBA{R: 150, A: 255})
	setRGBA(img, 3, 3, color.RGBA{R: 160, A: 255})

	redactFace(img, img.Bounds(), "pixel", 2)

	assertBlockColor(t, img, image.Rect(0, 0, 2, 2), color.RGBA{R: 10, A: 255})
	assertBlockColor(t, img, image.Rect(2, 0, 4, 2), color.RGBA{R: 30, A: 255})
	assertBlockColor(t, img, image.Rect(0, 2, 2, 4), color.RGBA{R: 90, A: 255})
	assertBlockColor(t, img, image.Rect(2, 2, 4, 4), color.RGBA{R: 110, A: 255})
}

func TestRedactFaceSecureUsesBorderAverage(t *testing.T) {
	img := newFilledImage(5, 5, color.RGBA{R: 5, G: 10, B: 15, A: 255})
	border := color.RGBA{R: 40, G: 90, B: 130, A: 255}

	for x := 1; x < 4; x++ {
		setRGBA(img, x, 0, border)
		setRGBA(img, x, 4, border)
	}
	for y := 1; y < 4; y++ {
		setRGBA(img, 0, y, border)
		setRGBA(img, 4, y, border)
	}
	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			setRGBA(img, x, y, color.RGBA{R: 200, G: 10, B: 20, A: 255})
		}
	}

	redactFace(img, image.Rect(1, 1, 4, 4), "secure", 4)

	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			if got := rgbaAt(img, x, y); got != border {
				t.Fatalf("pixel (%d,%d) = %+v, want border average %+v", x, y, got, border)
			}
		}
	}
}

func TestRedactFaceGaussBlursRegionOnly(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 5, 5))
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			fill := color.RGBA{R: 20, G: 20, B: 20, A: 255}
			if x >= 1 && x < 4 && y >= 1 && y < 4 {
				if (x+y)%2 == 0 {
					fill = color.RGBA{R: 0, G: 0, B: 0, A: 255}
				} else {
					fill = color.RGBA{R: 255, G: 255, B: 255, A: 255}
				}
			}
			setRGBA(img, x, y, fill)
		}
	}
	originalCenter := rgbaAt(img, 2, 2)
	originalOutside := rgbaAt(img, 0, 0)

	redactFace(img, image.Rect(1, 1, 4, 4), "gauss", 1)

	if got := rgbaAt(img, 0, 0); got != originalOutside {
		t.Fatalf("outside pixel changed: got %+v want %+v", got, originalOutside)
	}
	if got := rgbaAt(img, 2, 2); got == originalCenter {
		t.Fatalf("center pixel was not blurred: got %+v", got)
	}
	if got := rgbaAt(img, 2, 2); got.A != 255 {
		t.Fatalf("center alpha = %d, want 255", got.A)
	}
}

func TestParanoidTrackerApplyBlurAllBlacksEveryDetectedFace(t *testing.T) {
	tracker := &paranoidTracker{
		opts: &Options{BlurStrength: 2, BoxScale: 1.0},
	}
	frame := newFilledImage(6, 4, color.RGBA{R: 90, G: 100, B: 110, A: 255})

	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{0, 0, 2, 2}},
			{Loc: []int{3, 1, 5, 3}},
		},
		"blur-all",
		"black",
		false,
		false,
		0,
	)
	if err != nil {
		t.Fatalf("apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
	assertBlockColor(t, got, image.Rect(3, 1, 5, 3), color.RGBA{A: 255})
	if px := rgbaAt(got, 2, 0); px != (color.RGBA{R: 90, G: 100, B: 110, A: 255}) {
		t.Fatalf("non-face pixel changed: got %+v", px)
	}
}

func TestParanoidTrackerApplyBlurAllUsesScaledBoxes(t *testing.T) {
	tracker := &paranoidTracker{
		opts: &Options{BlurStrength: 2, BoxScale: 2.0},
	}
	frame := newFilledImage(6, 4, color.RGBA{R: 90, G: 100, B: 110, A: 255})

	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{2, 1, 4, 3}},
		},
		"blur-all",
		"black",
		false,
		false,
		0,
	)
	if err != nil {
		t.Fatalf("apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(1, 0, 5, 4), color.RGBA{A: 255})
	if px := rgbaAt(got, 0, 0); px != (color.RGBA{R: 90, G: 100, B: 110, A: 255}) {
		t.Fatalf("outside scaled box changed: got %+v", px)
	}
}

func TestParanoidTrackerApplyTargetedBlursOnlyMatchedFacesAndLingers(t *testing.T) {
	tracker := &paranoidTracker{
		targetVariants: []store.VariantData{
			{IdentityID: 7, Vec: []float64{1, 0}},
		},
		opts: &Options{
			MatchThreshold: 0.6,
			BlurStrength:   2,
			BoxScale:       1.0,
		},
	}

	frame1 := newFilledImage(6, 4, color.RGBA{R: 120, G: 130, B: 140, A: 255})
	out1, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame1.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{0, 0, 2, 2}, Vec: []float64{1, 0}},
			{Loc: []int{3, 0, 5, 2}, Vec: []float64{0, 1}},
		},
		"targeted",
		"black",
		false,
		false,
		1,
	)
	if err != nil {
		t.Fatalf("frame 1 apply returned error: %v", err)
	}

	got1 := rgbaImageFromBytes(out1, 6, 4)
	assertBlockColor(t, got1, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
	if px := rgbaAt(got1, 3, 0); px != (color.RGBA{R: 120, G: 130, B: 140, A: 255}) {
		t.Fatalf("non-target face was redacted: got %+v", px)
	}
	if len(tracker.activeTracks) != 1 || tracker.activeTracks[0].ID != 7 {
		t.Fatalf("unexpected active tracks after frame 1: %+v", tracker.activeTracks)
	}

	frame2 := newFilledImage(6, 4, color.RGBA{R: 120, G: 130, B: 140, A: 255})
	out2, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame2.Pix...),
		6,
		4,
		1,
		nil,
		"targeted",
		"black",
		false,
		false,
		1,
	)
	if err != nil {
		t.Fatalf("frame 2 apply returned error: %v", err)
	}

	got2 := rgbaImageFromBytes(out2, 6, 4)
	assertBlockColor(t, got2, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
}

func TestParanoidTrackerApplyTargetedLingersMultipleTrackedInstances(t *testing.T) {
	tracker := &paranoidTracker{
		targetVariants: []store.VariantData{
			{IdentityID: 7, Vec: []float64{1, 0}},
		},
		opts: &Options{
			MatchThreshold: 0.6,
			BlurStrength:   2,
			BoxScale:       1.0,
		},
	}

	frame1 := newFilledImage(6, 4, color.RGBA{R: 120, G: 130, B: 140, A: 255})
	if _, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame1.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{0, 0, 2, 2}, Vec: []float64{1, 0}},
			{Loc: []int{4, 0, 6, 2}, Vec: []float64{1, 0}},
		},
		"targeted",
		"black",
		false,
		false,
		1,
	); err != nil {
		t.Fatalf("frame 1 apply returned error: %v", err)
	}

	if got := len(tracker.activeTracks); got != 2 {
		t.Fatalf("expected 2 active tracks after frame 1, got %d", got)
	}

	frame2 := newFilledImage(6, 4, color.RGBA{R: 120, G: 130, B: 140, A: 255})
	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame2.Pix...),
		6,
		4,
		1,
		[]types.FaceResult{
			{Loc: []int{4, 0, 6, 2}, Vec: []float64{1, 0}},
		},
		"targeted",
		"black",
		false,
		false,
		1,
	)
	if err != nil {
		t.Fatalf("frame 2 apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
	assertBlockColor(t, got, image.Rect(4, 0, 6, 2), color.RGBA{A: 255})
	if px := rgbaAt(got, 2, 3); px != (color.RGBA{R: 120, G: 130, B: 140, A: 255}) {
		t.Fatalf("non-face pixel changed: got %+v", px)
	}
}

func TestParanoidTrackerApplyParanoidModeBlursDetectedFacesAndLastKnownTarget(t *testing.T) {
	tracker := &paranoidTracker{
		targetVariants: []store.VariantData{
			{IdentityID: 7, Vec: []float64{1, 0}},
		},
		opts: &Options{
			MatchThreshold: 0.6,
			BlurStrength:   2,
			BoxScale:       1.0,
		},
	}

	initial := newFilledImage(6, 4, color.RGBA{R: 80, G: 90, B: 100, A: 255})
	if _, err := tracker.apply(
		context.Background(),
		append([]byte(nil), initial.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{0, 0, 2, 2}, Vec: []float64{1, 0}},
		},
		"targeted",
		"black",
		false,
		false,
		2,
	); err != nil {
		t.Fatalf("seed frame apply returned error: %v", err)
	}

	next := newFilledImage(6, 4, color.RGBA{R: 80, G: 90, B: 100, A: 255})
	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), next.Pix...),
		6,
		4,
		1,
		[]types.FaceResult{
			{Loc: []int{3, 1, 5, 3}, Vec: []float64{0, 1}},
		},
		"targeted",
		"black",
		true,
		false,
		2,
	)
	if err != nil {
		t.Fatalf("paranoid frame apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(3, 1, 5, 3), color.RGBA{A: 255})
	assertBlockColor(t, got, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
}

func TestParanoidTrackerApplyParanoidModeTriggersWhenOneOfMultipleInstancesIsLost(t *testing.T) {
	tracker := &paranoidTracker{
		targetVariants: []store.VariantData{
			{IdentityID: 7, Vec: []float64{1, 0}},
		},
		opts: &Options{
			MatchThreshold: 0.6,
			BlurStrength:   2,
			BoxScale:       1.0,
		},
	}

	initial := newFilledImage(6, 4, color.RGBA{R: 80, G: 90, B: 100, A: 255})
	if _, err := tracker.apply(
		context.Background(),
		append([]byte(nil), initial.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{0, 0, 2, 2}, Vec: []float64{1, 0}},
			{Loc: []int{4, 0, 6, 2}, Vec: []float64{1, 0}},
		},
		"targeted",
		"black",
		false,
		false,
		2,
	); err != nil {
		t.Fatalf("seed frame apply returned error: %v", err)
	}

	next := newFilledImage(6, 4, color.RGBA{R: 80, G: 90, B: 100, A: 255})
	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), next.Pix...),
		6,
		4,
		1,
		[]types.FaceResult{
			{Loc: []int{4, 0, 6, 2}, Vec: []float64{1, 0}},
			{Loc: []int{2, 2, 4, 4}, Vec: []float64{0, 1}},
		},
		"targeted",
		"black",
		true,
		false,
		2,
	)
	if err != nil {
		t.Fatalf("paranoid frame apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(0, 0, 2, 2), color.RGBA{A: 255})
	assertBlockColor(t, got, image.Rect(4, 0, 6, 2), color.RGBA{A: 255})
	assertBlockColor(t, got, image.Rect(2, 2, 4, 4), color.RGBA{A: 255})
}

func TestParanoidTrackerApplyParanoidStrictBlursBeforeTargetSeen(t *testing.T) {
	tracker := &paranoidTracker{
		targetVariants: []store.VariantData{
			{IdentityID: 7, Vec: []float64{1, 0}},
		},
		opts: &Options{
			MatchThreshold: 0.6,
			BlurStrength:   2,
			BoxScale:       1.0,
		},
	}
	frame := newFilledImage(6, 4, color.RGBA{R: 150, G: 160, B: 170, A: 255})

	out, err := tracker.apply(
		context.Background(),
		append([]byte(nil), frame.Pix...),
		6,
		4,
		0,
		[]types.FaceResult{
			{Loc: []int{3, 1, 5, 3}, Vec: []float64{0, 1}},
		},
		"targeted",
		"black",
		true,
		true,
		2,
	)
	if err != nil {
		t.Fatalf("strict paranoid apply returned error: %v", err)
	}

	got := rgbaImageFromBytes(out, 6, 4)
	assertBlockColor(t, got, image.Rect(3, 1, 5, 3), color.RGBA{A: 255})
}

func newFilledImage(width, height int, fill color.RGBA) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			setRGBA(img, x, y, fill)
		}
	}
	return img
}

func setRGBA(img *image.RGBA, x, y int, c color.RGBA) {
	off := img.PixOffset(x, y)
	img.Pix[off] = c.R
	img.Pix[off+1] = c.G
	img.Pix[off+2] = c.B
	img.Pix[off+3] = c.A
}

func rgbaAt(img *image.RGBA, x, y int) color.RGBA {
	off := img.PixOffset(x, y)
	return color.RGBA{
		R: img.Pix[off],
		G: img.Pix[off+1],
		B: img.Pix[off+2],
		A: img.Pix[off+3],
	}
}

func rgbaImageFromBytes(pix []byte, width, height int) *image.RGBA {
	return &image.RGBA{
		Pix:    pix,
		Stride: width * 4,
		Rect:   image.Rect(0, 0, width, height),
	}
}

func assertBlockColor(t *testing.T, img *image.RGBA, rect image.Rectangle, want color.RGBA) {
	t.Helper()
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			if got := rgbaAt(img, x, y); got != want {
				t.Fatalf("pixel (%d,%d) = %+v, want %+v", x, y, got, want)
			}
		}
	}
}
