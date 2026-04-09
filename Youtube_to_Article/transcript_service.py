from langchain_community.document_loaders import YoutubeLoader
import yt_dlp
import requests

def get_transcript_loader(link: str) -> str:
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    return docs[0].page_content

# Method 2: yt-dlp fallback
# ---------------------------
def get_transcript_ytdlp(url: str) -> str:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        subs = info.get("subtitles") or info.get("automatic_captions")

        if not subs:
            raise Exception("No captions available")

        lang = "en" if "en" in subs else list(subs.keys())[0]
        subtitle_url = subs[lang][0]["url"]

        response = requests.get(subtitle_url)
        return response.text

# Fallback Wrapper
# ---------------------------
def extract_transcript(url: str) -> str:

    # ✅ Step 1: LangChain Loader
    try:
        return get_transcript_loader(url)

    except Exception as e1:
        print("YoutubeLoader failed → trying API...")

            # 🔁 Step 2: yt-dlp
        try:
            return get_transcript_ytdlp(url)

        except Exception as e3:
            raise Exception(
                f"All transcript methods failed:\n"
                f"Loader Error: {e1}\n"
                f"yt-dlp Error: {e3}"
            )
