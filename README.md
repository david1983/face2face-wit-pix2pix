# face2face with pix2pix

Commands needed to run everything


```
conda create --name tf15
conda install -c anaconda tensorflow-gpu=1.5
conda uninstall cudnn
conda install -c cudnn=7.1
pip install dlib opencv-python 

# rename your video as video.mp4 and copy it in the root folder
python generate_train_data.py --file video.mp4 --num 100 --landmark-model shape_predictor_68_face_landmarks.dat

mkdir pix2pix-tensorflow/photos
mv original pix2pix-tensorflow/photos
mv landmarks pix2pix-tensorflow/photos

cd pix2pix-tensorflow

./gensplit.sh

./train.sh

cd ..

./reduce.sh && ./freeze.sh

python run_webcam.py --source 0 --show 1 --landmark-model shape_predictor_68_face_landmarks.dat --tf-model face2face-reduced-model/frozen_model.


```
