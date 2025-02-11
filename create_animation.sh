#! /bin/zsh

if [ -z "$1" ]
then
  echo "No given filename"
  exit
fi

cd iterations
ffmpeg -f image2 -r 10 -pattern_type glob -i '*.png' -vcodec libx264 -crf 22 -pix_fmt yuv420p ../animations/$1.mp4
