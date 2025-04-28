# Vitpose_Pose_Prediction

Run test.py to test the current model with testing_poses.pkl. It should display all images with a 0.5 second pause in between.

To create a new model, run train.py which uses training_poses.pkl 

To extract additional 3DPW data, change the SEQUENCE_DIR and IMAGE_DIR paths for the training/testing data in extract_data.py. The IMAGE_DIR path should include the sequence images in the SEQUENCE_DIR path .pkl files.

Change the paths again when extracting testing data.