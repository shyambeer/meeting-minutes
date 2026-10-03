#!/bin/bash
# Builds sample_meeting.mp3 from sample_meeting_script.txt using macOS "say" + ffmpeg.
set -e
cd "$(dirname "$0")"
rm -rf parts && mkdir parts
: > parts/list.txt
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  name="${line%%:*}"
  text="${line#*: }"
  case "$name" in
    Priya)  v="Samantha" ;;
    Marcus) v="Daniel" ;;
    Elena)  v="Karen" ;;
    Sam)    v="Alex" ;;
    *)      v="Samantha" ;;
  esac
  i=$((i+1))
  n=$(printf %03d "$i")
  say -v "$v" -o "parts/$n.aiff" "$text" 2>/dev/null || say -o "parts/$n.aiff" "$text"
  ffmpeg -y -v error -i "parts/$n.aiff" -ac 1 -ar 22050 "parts/$n.wav"
  echo "file '$n.wav'" >> parts/list.txt
done < sample_meeting_script.txt

ffmpeg -y -v error -f concat -safe 0 -i parts/list.txt -codec:a libmp3lame -q:a 4 sample_meeting.mp3
rm -rf parts
echo "Created $(pwd)/sample_meeting.mp3"
