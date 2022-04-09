
from pytube import YouTube

def download_youtube(url, res='hi'):
    yt = YouTube(url)
    streams = yt.streams.filter(progressive=True, file_extension='mp4')
    ordered = streams.order_by('resolution')
    streams = ordered.desc() if res == 'hi' else streams.asc() # lowest first
    video = streams.first().download()
    return video