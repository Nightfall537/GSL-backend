#!/usr/bin/env python3
"""
Download GSL Videos from YouTube
Downloads curated Ghanaian Sign Language videos for training
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video_urls():
    """Load video URLs from gsl_video_urls.txt"""
    urls_file = Path("gsl_video_urls.txt")
    
    if not urls_file.exists():
        logger.warning("⚠️ gsl_video_urls.txt not found")
        return {}
    
    videos = {}
    current_category = 'general'
    
    with open(urls_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                # Check if it's a category header
                if line.startswith('# ') and 'Sign' in line:
                    current_category = line.replace('#', '').strip().lower().split()[0]
                continue
            
            # Parse URL | Category format
            if '|' in line:
                url, category = line.split('|', 1)
                url = url.strip()
                category = category.strip().lower()
            else:
                url = line.strip()
                category = current_category
            
            # Add to videos dict
            if category not in videos:
                videos[category] = []
            videos[category].append(url)
    
    return videos

# Default videos if file is empty
DEFAULT_VIDEOS = {
    'search_instructions': [
        '# Search YouTube for these terms:',
        '# - "Ghana Sign Language colors"',
        '# - "GSL family signs"', 
        '# - "Ghana Sign Language food"',
        '# - "GSL animals"',
        '# - "Ghana Sign Language numbers"',
        '# Then add URLs to gsl_video_urls.txt'
    ]
}

def install_ytdlp():
    """Install yt-dlp if not available"""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        logger.info("✅ yt-dlp already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("📦 Installing yt-dlp...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
            logger.info("✅ yt-dlp installed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to install yt-dlp: {e}")
            return False

def download_video(url, output_dir, category):
    """Download a single video from YouTube"""
    try:
        logger.info(f"📥 Downloading {category} video...")
        logger.info(f"🔗 URL: {url}")
        
        # Download with yt-dlp
        # Format: best video up to 1080p + best audio
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '--merge-output-format', 'mp4',
            '-o', str(output_dir / f'{category}_%(title)s.%(ext)s'),
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Downloaded {category} video")
            return True
        else:
            logger.error(f"❌ Failed to download {category}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading {category}: {e}")
        return False

def download_all_videos():
    """Download all GSL videos"""
    output_dir = Path("sam2_annotation/gsl_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load URLs from file
    GSL_VIDEOS = load_video_urls()
    
    if not GSL_VIDEOS:
        logger.warning("⚠️ No video URLs found in gsl_video_urls.txt")
        logger.info("\n📋 To add videos:")
        logger.info("1. Search YouTube for 'Ghana Sign Language' videos")
        logger.info("2. Copy video URLs")
        logger.info("3. Add them to gsl_video_urls.txt")
        logger.info("4. Run this script again")
        return False
    
    logger.info("🚀 Starting GSL Video Downloads from YouTube")
    logger.info("=" * 70)
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📊 Categories: {len(GSL_VIDEOS)}")
    logger.info(f"📹 Total videos: {sum(len(urls) for urls in GSL_VIDEOS.values())}")
    logger.info("=" * 70)
    
    downloaded = 0
    failed = 0
    
    for category, urls in GSL_VIDEOS.items():
        logger.info(f"\n📂 Category: {category.upper()}")
        logger.info(f"📹 Videos to download: {len(urls)}")
        
        for idx, url in enumerate(urls, 1):
            logger.info(f"\n[{idx}/{len(urls)}] Downloading {category}...")
            
            if download_video(url, output_dir, category):
                downloaded += 1
            else:
                failed += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Download Complete!")
    logger.info(f"✅ Successfully downloaded: {downloaded}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Videos location: {output_dir}")
    logger.info("=" * 70)
    
    if downloaded > 0:
        logger.info("\n🚀 Next step: python complete_sam2_pipeline.py")
    
    return downloaded > 0

def main():
    """Main function"""
    print("🎯 GSL YouTube Video Downloader")
    print("📥 Downloads curated GSL videos for training")
    print("=" * 70)
    
    # Install yt-dlp
    if not install_ytdlp():
        print("\n❌ Cannot proceed without yt-dlp")
        print("💡 Install manually: pip install yt-dlp")
        return
    
    # Download videos
    success = download_all_videos()
    
    if success:
        print("\n✅ Videos ready for training!")
    else:
        print("\n⚠️ No videos downloaded")
        print("💡 Check the URLs in the script")
        print("💡 You can add your own GSL video URLs")

if __name__ == "__main__":
    main()
