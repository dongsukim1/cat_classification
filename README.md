# Automated Wildlife Camera Trap Classifier

This project and writeup are both currently a work in progress. What is present may not be reflective of current progress. The dataset is derived from the LILA BC Caltech Camera Trap dataset originally published in "Recognition in Terra Incognita" for ECCV. The original publication created 
the dataset as a benchmark to test the ability of SOTA Computer Vision models to be able to generalize between locations. This project aims to use it to finetune a machine learning model to be able to recognize bobcat
camera traps with high accuracy and eliminate the need for manual classification. The original dataset contains 21 species but my dataset will only use images from 6 species. I designed it this way because it would best 
match the geographical distributions of interest. 

Data Collection

I collected the data by filtering our 6 species of interest from the total 243,000 camera trap images from an AWS s3 bucket into my own. My six classes were bobcats, coyotes, deers, "empty", fox, and mountain lions. I performed
preliminary exploratory data analysis found in EDA.py. The output is truncated here on Github so you'll have to download the repo if you want to see the full output. I picked out a couple random images from each class and immediately
noticed potential problems. Some images contained only partial captures of animals, others showed them in motion/blurry, and another sizable fraction were taken in the dark. It would be impossible to identify them with the human eye in 
some cases. There was also another issue in that I wasn't really aware what constituted "too much". When I worked on Computer Vision problems in biology the images typically were not as heavily distorted or irregular so I could simply filter
them out. I converted the images to grayscale and used that as my metric for "darkness" and I took the laplacian to measure how blurry they were. It turned out they weren't as problematic as expected. However, a more certain problem was the
heavy class imbalances. The dataset only contained <200 empty images. I had to augment the dataset using empty images from a similar dataset. I initially ran a simple CNN training run on the problematic dataset and after augmentation the accuracy
improved for all classes by ~10% using the same architecture. 

Citation:
Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. Proceedings of the 15th European Conference on Computer Vision (ECCV 2018). 
