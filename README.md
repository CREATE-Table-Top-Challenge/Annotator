# Automated_Annotator
Make sure to install torch first. All the (satisfied) libraries should already be installed if you install the remaining libraries. 

Libraries needed:  
- torch (https://pytorch.org/get-started/locally/)
- ultralytics 
- torchvision (satisfied)
- transformers  
- PyQt6
- superqt
- sklearn
- opencv-python (satisfied)
- PIL (satisfied)
- yaml (satisfied)
- numpy (satisfied)
- pandas (satisfied)

To run type command in console: python \<path to repo\>/Automated_Annotator.py  

Commands:
- n -> move to next image
- p -> move to previous image
- d -> remove image (only functional after all images have been labelled)
- m -> cycle display modes (show all boxes, show active box only, hide all boxes)
- Ctrl+e -> export bounding boxes to label file (only functional after all images have been labelled)
