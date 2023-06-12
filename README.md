# Test_Script_LOVEU_Track4
Official Test Script for LOVEU@CVPR 2023 Track 4 Text-Guided Video Editing

## Introduction
This repository contains the official test script for LOVEU@CVPR 2023 Track 4 Text-Guided Video Editing. The test script is used to evaluate the performance of the submitted videos on three evaluation metrics: 1)TextualAlignmentScore, 2)FrameConsistencyScore, and 3)PickScore. The detailed description of the evaluation metrics can be found in the [Track 4 webpage](https://sites.google.com/view/loveucvpr23/track4?authuser=0).

## Usage
The video files should be organized the same as Demo_Folder. Below is the example of the video files organization:

```
Demo_Folder
├── DAVIS_480p
│   ├── bmx-rider
|   |   ├── background
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   ├── ...
│   │   │   ├── A person does a trick on a BMX bike at a snow covered park in Antarctica..gif
│   │   ├── multiple
│   │   │   ├── ...
│   │   ├── object
│   │   │   ├── ...
│   │   ├── style
│   │   │   ├── ...
│   ├── ...
├── videvo_480p
│   ├── ...
├── youtube_480p
│   ├── ...
├── LOVEU-TGVE-2023_Dataset.csv
├── cal.py
├── transform.py
```

Besides, video names should be the same as video_caption.gif. The video_caption can be found in LOVEU-TGVE-2023_Dataset.csv. 

To run the test script, please use the following command:

```
python cal.py --path [path to the folder containing the video files] --device [device name]
```

Besides, if your video files are organized in the jpg format, please use the following command to convert them to gif format:

```
python transform.py --path [path to the folder containing the video files]
```

## Useful Links
- [Track 4 Webpage](https://sites.google.com/view/loveucvpr23/track4?authuser=0)
- [Track 4 Dataset](https://drive.google.com/file/d/1D7ZVm66IwlKhS6UINoDgFiFJp_mLIQ0W/view)

## Miscellaneous
If you have any questions, please contact us at <jinbin5bai@gmail.com>.