.PHONY: help fix-rotation

# Default target
help:
	@echo "Video Editor Commands:"
	@echo "  make fix-rotation INPUT=video.MOV [OUTPUT=video-fixed.mp4] [TRANSPOSE=1] [METADATA_ONLY=1]"
	@echo "  make fix-rotation ARGS='video.MOV [video-fixed.mp4]' [TRANSPOSE=1] [METADATA_ONLY=1]"
	@echo "  (omit OUTPUT → writes <input_stem>_fixed.mp4)"
	@echo ""
	@echo "METADATA_ONLY=1: Only remove rotation metadata (instant, no re-encoding)"
	@echo "                 Use if frames are already correctly oriented"
	@echo ""
	@echo "TRANSPOSE values (only used if METADATA_ONLY is not set):"
	@echo "  1 = 90° clockwise (for -90° rotation, most common iPhone portrait)"
	@echo "  2 = 90° counterclockwise (for 90° rotation)"
	@echo "  3 = 180° (for 180° rotation)"

# Fix video rotation. OUTPUT defaults to <input_stem>_fixed.mp4 (always MP4).
# Usage: make fix-rotation INPUT=video.MOV [OUTPUT=video-fixed.mp4] [METADATA_ONLY=1] [TRANSPOSE=1]
#    or: make fix-rotation ARGS='video.MOV [video-fixed.mp4]' [METADATA_ONLY=1] [TRANSPOSE=1]
# METADATA_ONLY=1: Only remove rotation metadata (instant, no re-encoding)
# TRANSPOSE=1 is default (90° clockwise, fixes -90° iPhone portrait videos)
fix-rotation:
	@INPUT_VAL=$$([ -n "$(INPUT)" ] && echo "$(INPUT)" || echo "$(word 1,$(ARGS))"); \
	OUTPUT_VAL=$$([ -n "$(OUTPUT)" ] && echo "$(OUTPUT)" || echo "$(word 2,$(ARGS))"); \
	if [ -z "$$OUTPUT_VAL" ]; then \
		OUTPUT_VAL="$$(dirname "$$INPUT_VAL")/$$(basename "$${INPUT_VAL%.*}")_fixed.mp4"; \
	fi; \
	METADATA_ONLY_VAL=$$([ -n "$(METADATA_ONLY)" ] && echo "$(METADATA_ONLY)" || echo "0"); \
	TRANSPOSE_VAL=$$([ -n "$(TRANSPOSE)" ] && echo "$(TRANSPOSE)" || echo "1"); \
	if [ -z "$$INPUT_VAL" ]; then \
		echo "Error: INPUT must be set"; \
		echo "Usage: make fix-rotation INPUT=video.MOV [OUTPUT=video-fixed.mp4] [METADATA_ONLY=1] [TRANSPOSE=1]"; \
		echo "   or: make fix-rotation ARGS='video.MOV [video-fixed.mp4]' [METADATA_ONLY=1] [TRANSPOSE=1]"; \
		echo "  If OUTPUT is omitted, writes <input_stem>_fixed.mp4"; \
		exit 1; \
	fi; \
	if [ "$$METADATA_ONLY_VAL" = "1" ]; then \
		echo "Removing rotation metadata only (instant) for $$INPUT_VAL -> $$OUTPUT_VAL..."; \
		ffmpeg -y -i "$$INPUT_VAL" -c copy -metadata:s:v:0 rotate=0 "$$OUTPUT_VAL" 2>&1; \
	else \
		echo "Fixing rotation (transpose=$$TRANSPOSE_VAL, requires re-encoding) for $$INPUT_VAL -> $$OUTPUT_VAL..."; \
		ffmpeg -y -hwaccel videotoolbox -i "$$INPUT_VAL" -vf "transpose=$$TRANSPOSE_VAL" -map_metadata -1 -c:v h264_videotoolbox -b:v 12M -c:a copy -stats_period 1 "$$OUTPUT_VAL" 2>&1; \
	fi
