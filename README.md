# Machine Learning on Bone (re)Modeling

This project contains the code and data for the poster "An application of neural networks to identifying cellular bone growth processes" by Shannon P. McPherron, Philipp Gunz and Alexandra Schuh presented at the 2024 ESHE meetings in Zagreb, Croatia.  The poster is included here in the folder eshe_2024_poster.  The folder unet contains code for two unet models, one on the images themselves and one on textures computed from the images.  The folder yolo contains code for a YOLOv8 model.

The unet and yolo folders are kept separate because they have different Python dependencies.  A requirements.txt file is provided for each folder.

The authors reserve the right to change this code after the meetings in order to improve it.  However, a branch will be made to preserve this version of the code.

Note that the original TIFF images are not included in this repository.  Downsampled versions of these images are included in yolo and unet.  After the meetings, this structure will be changed to have one copy of the images at YOLOv8 resolution which can be further downsampled for the unet models.
