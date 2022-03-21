from pytube import YouTube
YouTube('https://youtu.be/2lAe1cqCOXo').streams.first().download()

yt = YouTube('http://youtube.com/watch?v=2lAe1cqCOXo')
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

def download_youtube(url, res='hi'):
    streams = yt.streams.filter(progressive=True, file_extension='mp4')
    ordered = streams.order_by('resolution')
    streams = streams.desc() if res == 'hi' else streams.asc() # lowest first
    video = streams.first().download()