# Egocentric Video Segmentation
Final Year Honours Project - Software used to segment a first-person video into Transitions and Visits

# Requirements
* Install [Python 3.8.2](https://www.python.org/downloads/release/python-382/)
* Install [pip](https://pip.pypa.io/en/stable/installing/) (or your own package manager)
    * Please run ```pip install -r requirements.txt``` in order to install the necessary requirements, or again through your own package manager.
* [OpenCV 4.2.0](https://opencv.org/releases/)

# OpticalFlow
## How to use
Using Powershell/Command Prompt, run:
```powershell
python of.py "./Video/Path/Here.mp4"
```
Which will then output a file called ```opticalflow_results.csv``` to be used in the Classifier script.

## Screenshots
![Screenshot of OpticalFlow run from PowerShell (on a 40 minute video)](./ReadMe_Assets/OpticalFlow_CMD_Screenshot.png?raw=true "Screenshot of OpticalFlow run from PowerShell (on a 40 minute video)")
Note: Time stated above is for a 40 minute video, your time may vary depending on computer specifications!
![Screenshot of OpticalFlow window](./ReadMe_Assets/OpticalFlow_Screenshot.png?raw=true "Screenshot of OpticalFlow window")


# Classifier
## How to use
Using Powershell/Command Prompt, run:
```powershell
python analysis.py "TrainDataset" "GroundTruth_Train" "TestDataset" "OutputFile"
```

Where:
* **TrainDataset**: The optical flow file for the training data used to train the model (e.g. opticalflow.csv)
* **GroundTruth_Train**: The GroundTruth Data for the supplied optical flow file (.EAF annotated file e.g. annotate.eaf)
* **TestDataset**: The optical flow file for the video you wish to be annotated (test_opticalflow.csv)
* **OutputFile**: The name and/or location of the file you wish to be outputted (a .EAF file e.g. output.eaf)

**Note: Please ensure you have the ```template.eaf``` file downloaded and in the same directory when running the script!**

## Screenshots
![Screenshot of Analysis run from PowerShell](./ReadMe_Assets/Analysis_Screenshot.png?raw=true "Screenshot of OpticalFlow run from PowerShell")

# Using the Data
Once both of these scripts have run successfully, you will be left with a file ```YourNameHere.eaf```. Open this with the program [ELAN](https://archive.mpi.nl/tla/elan) to view the completed annotated video. You will need to link the location of the video within the application in order to view it correctly.

To do this:
* Open ```YourFile.eaf``` in ELAN
* Press ***Edit*** then ***Linked Files...*** (or use the key combination ***CTRL+ALT+L***)
* If there are any files listed under the ***Linked Media Files*** tab, remove them and click **Apply**
* Press **Add** and select your video file
* **Apply** then **Close**

![Screenshot of ELAN Window](./ReadMe_Assets/ELAN_Screenshot.png?raw=true "ELAN Window Screenshot")
