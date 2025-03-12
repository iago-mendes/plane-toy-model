#! /bin/zsh

if [ -z "$1" ]
then
  echo "No given filename"
  exit
fi

cd iterations
ffmpeg -f image2 -r 2 -pattern_type glob -i "physical_commutator-*.png" -vcodec libx264 -crf 22 -pix_fmt yuv420p \
       ../animations/physical_commutator-$1.mp4
ffmpeg -f image2 -r 2 -pattern_type glob -i "physical_dyad_vectors-*.png" -vcodec libx264 -crf 22 -pix_fmt yuv420p \
       ../animations/physical_dyad_vectors-$1.mp4
