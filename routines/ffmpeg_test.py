import os
#values (29, 35, 41, 47) and maximum bitrate values
#(2Mb/s, 1.5Mb/s, 1Mb/s) are selected to create a to-
#tal of 12 data variants.
"""
Now, lets up all this knowledge to convert a single high quality file you have shot from your video camera and convert it into a low quality file, say flv for a player. Although the flags and the method we are using is all that matters, I am using flv to just define the output video's specs.

Video Bitrate: < 500 kbps

aspect ratio: 480x360

audio bitrate: 32kbps

Frames per second: 25

We will use the following command to make this happen.

[shredder12]$ ffmpeg -i  recorded_file.mov  -ar 22050  -ab 32k  -r 25  -s 480x360  -vcodec flv -qscale 9.5 output_file.avi

ar is used to set the audio frequency of the output file. The default value is 41000Hz but we are using a low value to produce a low flv quality file.

qscale is a quantisation scale which is basically a quality scale for variable bitrate and coding, with lower number indicating a higher quality. You can try running the above command with and without qscale flag and then you can easily see the quality difference.

In order to get a list of all the formats supported by ffmpeg, run this command.

[shredder12]$ ffmpeg -formats
"""
os.system("ffmpeg -i ./datasets/vehicles/Dense_LISA_1/Dense/jan28.avi -s 480x360 -qscale 47 -r 20 -crf 22 -codec flv ./datasets/output.avi")