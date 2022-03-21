#!/bin/bash
#source ~/anaconda3/bin/activate;
#conda create --yes --name FastStyleTransfer python=3.9;
#pip install --upgrade pip;
#pip install tensorflow matplotlib tensorflow_hub;
#conda activate FastStyleTransfer;
 
# source ~/anaconda3/bin/activate;
# conda activate FastStyleTransfer;
 
input_videos="./input/videos/*";
input_styles="./input/styles/*";
input_frames="./input/frames";
input_audio="./input/audio";
output_frames="./output/frames";
output_videos="./output/videos";
 
# Loop on each video in the input folder.
for video in $input_videos;
do
  echo $video;
  videoname=$(basename $video);
 
  # Extract all frames from the video file and save them in a new folder using 8-digit numbers with zero padding in an incremental order.
  input_frames_folder="$input_frames/$videoname";
  mkdir -p "$input_frames_folder";
  echo "extracting video frames from"
  echo $video
  ffmpeg -v verbose -i $video "$input_frames_folder/%08d.ppm";
 
  # Extract the audio file from the video to the format of an mp3. We will need this audio later to add it to the final product.
  input_audio_folder="$input_audio/$videoname";
  mkdir -p "$input_audio_folder";
 
  audio="";
  # Only VP8 or VP9 or AV1 video and Vorbis or Opus audio and WebVTT subtitles are supported for WebM.
  if [[ $videoname == *.webm ]]; then
    audio="$input_audio_folder/$videoname.ogg";
    ffmpeg -v quiet -i "$video" -vn -c:a libvorbis -y "$audio"; 
  else
    audio="$input_audio_folder/$videoname.mp3";
    ffmpeg -v quiet -i "$video" -vn -c:a libmp3lame -y "$audio";
  fi
 
  # Retrieve the frame rate from the input video. We will need it to configure the final video later.
  frame_rate=`ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate $video`;
 
  # Loop on each image style from the input styles folder.
  for style in $input_styles;
  do
    echo "$style";
    stylename=$(basename $style);
    output_frames_folder="$output_frames/$videoname/$stylename";
    mkdir -p "$output_frames_folder";
 
    # Stylize all frames using the input image and write all processed frames to the output folder.
    python3 faster.py --input "$input_frames_folder" --output "$output_frames_folder" --style "$style";
 
    # Combine all stylized video frames and the exported audio into a new video file.
    output_videos_folder="$output_videos/$videoname/$stylename";
    mkdir -p "$output_videos_folder";
    ffmpeg -v quiet -framerate "$frame_rate" -i "$output_frames_folder/%08d.ppm" -i "$audio" -pix_fmt yuv420p -acodec copy -y "$output_videos_folder/$videoname";
     
    rm -rf "$output_frames_folder";
  done
  rm -rf "$output_frames/$videoname";
  rm -rf "$input_frames_folder";
  rm -rf "$input_audio_folder";
done