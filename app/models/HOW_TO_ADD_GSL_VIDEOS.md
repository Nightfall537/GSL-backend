# How to Add GSL Training Videos from YouTube

## 🎯 Quick Start

### Step 1: Find GSL Videos on YouTube

Search for these terms on YouTube:
- `Ghana Sign Language colors`
- `GSL family signs`
- `Ghana Sign Language food`
- `GSL animals`
- `Ghana Sign Language numbers`
- `GSL greetings`
- `Ghana Sign Language alphabet`

### Step 2: Copy Video URLs

1. Find a relevant GSL video
2. Click the video
3. Copy the URL from your browser
   - Example: `https://www.youtube.com/watch?v=ABC123XYZ`

### Step 3: Add URLs to File

1. Open `gsl_video_urls.txt`
2. Paste the URL under the appropriate category
3. Save the file

Example:
```
# Colors
https://www.youtube.com/watch?v=8VqKx-5gXRY

# Family Signs
https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE

# Food Signs
https://www.youtube.com/watch?v=ANOTHER_VIDEO_ID
```

### Step 4: Download Videos

Run the download script:
```bash
python download_gsl_youtube_videos.py
```

This will:
- Install yt-dlp (if needed)
- Download all videos from your list
- Save them to `sam2_annotation/gsl_videos/`
- Ready for training!

### Step 5: Train Model

After downloading videos:
```bash
python complete_sam2_pipeline.py
```

---

## 📋 Recommended GSL YouTube Channels

Search these channels for quality GSL content:
- Ghana National Association of the Deaf
- GSL Learning channels
- Deaf education channels in Ghana

---

## 🎓 Tips

### Good Training Videos:
✅ Clear view of signer
✅ Good lighting
✅ Full body visible
✅ Multiple examples of same sign
✅ Slow, clear demonstrations

### Avoid:
❌ Poor lighting
❌ Signer too far away
❌ Only hands visible (need full body!)
❌ Too fast
❌ Low quality

---

## 🔧 Troubleshooting

### "yt-dlp not found"
```bash
pip install yt-dlp
```

### "No videos downloaded"
- Check your internet connection
- Verify URLs are correct
- Make sure URLs are YouTube links

### "Download failed"
- Some videos may be restricted
- Try a different video
- Check if video is still available

---

## 📊 Current Status

You already have:
- ✅ 1 colors video (trained)
- ✅ Working model (11 color gestures)
- ✅ Full-body tracking system

Add more videos to expand your model!

---

## 🚀 Quick Example

1. Search YouTube: "Ghana Sign Language family"
2. Find video, copy URL
3. Edit `gsl_video_urls.txt`:
   ```
   # Family Signs
   https://www.youtube.com/watch?v=YOUR_VIDEO_HERE
   ```
4. Run: `python download_gsl_youtube_videos.py`
5. Run: `python complete_sam2_pipeline.py`
6. Done! Model now knows family signs too!

---

**Need help?** The script will guide you through each step!
