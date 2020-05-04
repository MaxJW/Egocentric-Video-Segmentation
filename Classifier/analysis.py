# Packages
import numpy as np # Numpy package
import csv # Used to read in CSV files
# Packages for SVM Implementation
from sklearn import svm
from sklearn.metrics import accuracy_score
# Used to read and write EAF (XML) files for ELAN
import xml.etree.ElementTree as ET 
# Used to plot points from optical flow files
import matplotlib.pyplot as plt
# Used to check path exists for starting program
import os.path
from os import path
# Used to calculate time taken to complete script
from datetime import datetime
# Used to save model to file to import/export later
import pickle

#TrainDataset = "../OpticalFlow/opticalflow_result_video1.csv"
#GroundTruth_Train = "../Recordings/GroundTruth/video1_MAN.eaf"

class Analysis:
    def singlex(self, path): # Not used - Called to only produce results for one point in the video (i.e. (1,1))
        #Import CSV File
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            displacement = []
            curr_row = 0
            for row in csv_reader:
                if (curr_row % 1000 == 0):
                    print(curr_row)
                for traj in range(0, 3, 3):
                    if (int(row[traj]) == 1):
                        if (float(row[traj+1]) < 40 and float(row[traj+1]) > -40):
                            displacement.append(float(row[traj+1]))
                        else:
                            displacement.append(float(row[traj+1]))
                    else:
                        displacement.append(float(0))
                curr_row += 1
            print ("Processed: " + str(curr_row) + " rows")

        cd = []
        print("Calculating point: 1")
        cd.append(displacement[0])
        for i in range(1,len(displacement)):
            cd.append(cd[i-1] + displacement[i])

        plt.plot(cd)
        plt.show()
        return cd

    def multiplex(self, path): # Import all X coordinates of each point
        #Import CSV File
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            displacement = [[0 for x in range(0)] for y in range(50)]
            curr_row = 0
            for row in csv_reader:
                curr_col = 0
                if (curr_row % 10000 == 0):
                    print(curr_row)
                columns = len(row)
                for traj in range(0, columns, 3):
                    curr_col_data = displacement[curr_col]
                    if (int(row[traj]) == 1):
                        if (float(row[traj+1]) < 40 and float(row[traj+1]) > -40):
                            curr_col_data.append(float(row[traj+1]))
                        else:
                            curr_col_data.append(float(row[traj+1]))
                    else:
                        curr_col_data.append(float(0))
                    curr_col += 1
                curr_row += 1
            print ("Processed: " + str(curr_row) + " rows")

        print(len(displacement))
        print(len(displacement[0]))

        final_cd = []

        for points in range(0,len(displacement)):
            print("Calculating point: " + str(points+1))
            cd = []
            cd.append(displacement[points][0])
            count = 0
            for i in range(1,len(displacement[points])):
                cd.append(cd[count] + displacement[points][i])
                count += 1
            #plt.plot(cd)
            final_cd.append(cd)

        #plt.show()
        return final_cd #Return the total added points for the graph visual
    
    def all_xy(self, path): # Import all X and Y coordinates for each point in the video
        #Import CSV File
        with open(str(path)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            displacement = [[0 for x in range(50)] for y in range(0)]
            curr_row = 0
            for row in csv_reader:
                curr_col = 0
                if (curr_row % 10000 == 0):
                    print("Processed: " + str(curr_row) + " rows\r",end="")
                columns = len(row)
                curr_col_data = []
                for traj in range(0, columns, 3):
                    curr_col_data.append([float(row[traj+1]),float(row[traj+2])])
                displacement.append(curr_col_data)
                curr_row += 1
            print ("Processed: " + str(curr_row) + " rows")

        return displacement, curr_row

    def elanImport(self, path): # Import Ground Truth file for supplied video
        # Import file
        EAF_file = ET.parse(str(path)) # Import EAF file for Ground Truth
        ELAN = EAF_file.getroot() # Get root
        HEADER = ELAN.find('HEADER') # Store HEADER
        TIME_ORDER = ELAN.find('TIME_ORDER') # Store TIME_ORDER
        TIER = ELAN.find('TIER') #Store TIERs (transitions/visits)

        #HEADER
        time_units = HEADER.get('TIME_UNITS') #Not actually used, should be used for assuring milliseconds is used in EAF file
        
        #TIME_ORDER
        time_order = [] #Stores the frame numbers converted from milliseconds
        for slot in TIME_ORDER.iter('TIME_SLOT'):
            ts_id = slot.get('TIME_SLOT_ID')
            ts_value = slot.get('TIME_VALUE') #milliseconds
            frame_num = round(int(ts_value)/(1000/25)+1) #Convert from milliseconds to FPS (25 should be changed to a value found from the video file)
            time_order.append(frame_num)

        #TIER
        tier = [] #include startframe, endframe, and annotation (visit/transition)
        for annotation in TIER.iter('ANNOTATION'):
            curr_tier = []
            for time_slot in annotation.iter('ALIGNABLE_ANNOTATION'):
                curr_tier.append(time_order[int((time_slot.get('TIME_SLOT_REF1'))[2:])-1])
                curr_tier.append(time_order[int((time_slot.get('TIME_SLOT_REF2'))[2:])-1])
                for tag in time_slot.iter('ANNOTATION_VALUE'):
                    curr_tier.append(str(tag.text))
            tier.append(curr_tier)

        print("ELAN File Imported")

        return time_order, tier
    
    # Reshape and collect data to insert into SVM for training
    def SVMPreparer(self, tiers, total_frames):
        print("---| Generating Tag List |---")
        tier = 0 # Current annotation tier
        reset = 0
        annotate_num = int(tiers[tier][1]) # Contains end frame number
        annotate_tag = tiers[tier][2] # Contains Transition or Visit

        tv = [] # Holds transitions+visits (copy of cdc, used in case of skipping frames in below loop)
        tag_list = [] # Holds 1 or 0 depending on Transition or Visit respectively
        for curr in range(0, total_frames): # For each frame in the video, generate list of tags (i.e. 1 or 0 for each frame)
            progress = curr - reset
            if (progress % 10000 == 0): # Output progress every 10000 frames
                print("Progress: " + str(progress) + "\r",end="")
            if (progress < annotate_num): # If current frame is less than end frame, continue
                if (annotate_tag == "Transition"):
                    tag_list.append(1)
                else:
                    tag_list.append(0)
            else: # Otherwise, move to next tier and get new tag
                tier += 1
                if (tiers[tier][0] == '1'):
                    reset += annotate_num
                annotate_num = int(tiers[tier][1])
                annotate_tag = tiers[tier][2]
                if (annotate_tag == "Transition"):
                    tag_list.append(1)
                else:
                    tag_list.append(0)
        print("Prepared: " + str(total_frames) + " frames")

        return tag_list # Return final array and list of Transitions/Visits

    def SVMClassifier(self, dataset, ground_truth): # Create SVM Model
        print("---| Generating Model (May take a minute or two) |---")
        dataset = np.array(dataset)
        #print(tv.shape)
        dataset = np.reshape(dataset, (dataset.shape[0], -1)) # Reshape array for SVM input
        clf = svm.SVC() # New SVM Model (insert verbose=True for output during creation)
        clf.fit(dataset, ground_truth) # Fit Video1 (in this case) with a tag list generated from the ELAN file (where 1 is transition and 0 is visit)

        print("---| Generate Accuracy of Model (Compared against the training data) |---")
        predictions = clf.predict(dataset) # Predict using the same dataset
        print(accuracy_score(ground_truth, predictions)) # Compare the predicitions generated to the ground-truth tag_list (same as before)

        # Used for running multiple tests, testing purposes only!
        """ result = 0.0
        accuracy = []
        print(Tests)
        for i in Tests:
            of_csv = "../OpticalFlow/opticalflow_result_video" + str(i) + ".csv"
            man_eaf = "../Recordings/GroundTruth/video" + str(i) + "_MAN.eaf"
            fin_eaf = "../Recordings/video" + str(i) + "_CODE.eaf"
            test = Analysis().testSVM(clf, of_csv, man_eaf, fin_eaf)
            accuracy.append(test)
            result = result + test
        result = result/len(Tests)
        print(accuracy)
        print("Total Accuracy: " + str(result)) """

        return clf

    def testSVM(self, clf, dataset, ground_truth, output_file):
        print("---| Generate Predictions from " + dataset + " |---")
        print("---| Importing new optical flow file |---")
        new_ds, total_frames = Analysis().all_xy(dataset) # Bring in new test dataset
        print("---| Generating Predictions |---")
        new_ds = np.array(new_ds) # Convert to array
        new_ds = np.reshape(new_ds, (new_ds.shape[0], -1)) # Reshape to predict with SVM model
        video_predict = clf.predict(new_ds) #Predict using the same dataset

        accurate = 0
        if ground_truth:
            print("---| Import Ground Truth for Test Video |---")
            _, elan_test = Analysis().elanImport(ground_truth)
            print("---| Prepare Ground Truth for SVM |---")
            gt_test = Analysis().SVMPreparer(elan_test, total_frames)
            print("---| Accuracy for Test Video with Ground Truth |---")
            accurate = accuracy_score(gt_test, video_predict)
            print(accurate)

        # Output each frame of change (i.e. from Visit to Transition)
        curr = video_predict[0]
        framechange = []
        errors = 0
        #print("0 - " + str(curr))
        for i in range(total_frames):
            if (curr != video_predict[i]):
                framechange.append(i)
                curr = video_predict[i]
                #print(str(i) + " - " + str(curr))

        print("Number of Visits + Transitions: " + str(len(framechange))) # Output number of changes occuring
        return framechange, video_predict

    def elanExport(self, timeslots, prediction, output_file): # Export to ELAN file to compare and scroll through
        # Import file
        EAF_file = ET.parse(path.join(path.dirname(__file__), "./template.eaf")) # Import "empty" ELAN file
        ELAN = EAF_file.getroot() # Get root
        HEADER = ELAN.find('HEADER') # Store HEADER
        TIME_ORDER = ELAN.find('TIME_ORDER') # Store TIME_ORDER
        TIER = ELAN.find('TIER') #Store TIERs (transitions/visits)

        #HEADER
        MEDIA_DESCRIPTOR = HEADER.find('MEDIA_DESCRIPTOR')
        #MEDIA_DESCRIPTOR.set('MEDIA_URL',MEDIA_URL) # Full path for video (not sure if needed?)
        #MEDIA_DESCRIPTOR.set('RELATIVE_MEDIA_URL',RELATIVE_MEDIA_URL) # Relative path

        #TIME_ORDER
        to_id = "ts0" # Time order id starts at 0 with a timeslot of 0
        attrib = {'TIME_SLOT_ID': str(to_id), 'TIME_VALUE': str(0)}
        newslot = TIME_ORDER.makeelement('TIME_SLOT', attrib)
        TIME_ORDER.append(newslot) # Append default timeslot
        for count, frame in enumerate(timeslots):
            milli = (frame/25)*1000 #(frames/fps)*1000 to convert back to milliseconds
            to_id = "ts" + str(count+1)
            attrib = {'TIME_SLOT_ID': str(to_id), 'TIME_VALUE': str(int(milli))}
            newslot = TIME_ORDER.makeelement('TIME_SLOT', attrib)
            TIME_ORDER.append(newslot)

        #TIER
        # IDs for Transitions and Visits
        curr = int(prediction[0])
        # IDs drawn from the Controlled Vocabulary "TransitionsCV.ecv"
        transition = "cveid_f6b20d03-65c3-40a5-a7bf-0faa7b6e0fd4"
        visit = "cveid_d5ec9bd3-7fe0-48fe-9ccf-e927d9853929"
        for count, frame in enumerate(timeslots): # For each frame change, recreate annotation format and append to TIER list
            ANNOTATION = TIER.makeelement('ANNOTATION', {})
            aid = "a" + str(count)
            tsref1 = "ts" + str(count)
            tsref2 = "ts" + str(count+1)
            if (curr == 1):
                attrib = {'ANNOTATION_ID':str(aid),'CVE_REF':transition,'TIME_SLOT_REF1':str(tsref1),'TIME_SLOT_REF2':str(tsref2)}
            else:
                attrib = {'ANNOTATION_ID':str(aid),'CVE_REF':visit,'TIME_SLOT_REF1':str(tsref1),'TIME_SLOT_REF2':str(tsref2)}
            ALIGNABLE_ANNOTATION = ANNOTATION.makeelement('ALIGNABLE_ANNOTATION', attrib)
            ANNOTATION_VALUE = ALIGNABLE_ANNOTATION.makeelement('ANNOTATION_VALUE', {})
            if (curr == 1): # For each segment, switch to next tag (as i.e. a visit cannot be followed by another visit)
                ANNOTATION_VALUE.text = "Transition"
                curr = 0
            else:
                ANNOTATION_VALUE.text = "Visit"
                curr = 1
            ALIGNABLE_ANNOTATION.append(ANNOTATION_VALUE)
            ANNOTATION.append(ALIGNABLE_ANNOTATION)
            TIER.append(ANNOTATION)

        # Finish and write to file
        EAF_file.write(output_file)

def main():
    import sys
    startTime = datetime.now()
    dirname = path.dirname(__file__)
    EAFTemplate = "./template.eaf"

    try:
        TrainDataset = sys.argv[1]
        GroundTruth_Train = sys.argv[2]
        TestDataset = sys.argv[3]
        OutputFile = sys.argv[4]
    except:
        print('---| ERROR: Not all supplied arguments found! |---')
        print("Please supply 4 arguments in the following order:")
        print(" - TrainDataset : The optical flow file for the training data (e.g. opticalflow.csv)")
        print(" - GroundTruth_Train : The GroundTruth Data for the supplied optical flow file (.EAF annotated file e.g. annotate.eaf)")
        print(" - TestDataset : The optical flow file for the video you wish to be annotated (test_opticalflow.csv)")
        print(" - OutputFile : The name and/or location of the file you wish to be outputted (a .EAF file e.g. output.eaf)")
        sys.exit()

    if not path.exists(str(TrainDataset)):
        TrainDataset = path.join(dirname, str(TrainDataset))
        if not path.exists(TrainDataset):
            print('---| ERROR: Training Dataset "' + TrainDataset + '" not found! |---')
            sys.exit()
    
    if not path.exists(str(GroundTruth_Train)):
        GroundTruth_Train = path.join(dirname, str(GroundTruth_Train))
        if not path.exists(GroundTruth_Train):
            print('---| ERROR: Ground Truth Data "' + GroundTruth_Train + '" not found! |---')
            sys.exit()

    if not path.exists(str(TestDataset)):
        TestDataset = path.join(dirname, str(TestDataset))
        if not path.exists(TestDataset):
            print('---| ERROR: Test Dataset "' + TestDataset + '" not found! |---')
            sys.exit()
    
    if not path.exists(str(EAFTemplate)):
        EAFTemplate = path.join(dirname, str(EAFTemplate))
        if not path.exists(EAFTemplate):
            print(path.join(dirname, str(EAFTemplate)))
            print('---| ERROR: EAF Template file not found! |---')
            print('Please ensure "template.eaf" exists in the same directory as this script!')
            sys.exit()
    
    OutputFile = path.join(dirname, str(OutputFile))

    print('---| Resulting EAF file will be output to: "' + OutputFile + '" |---')
    testing = 0 #1 for testing
    if not path.exists(path.join(dirname, "trained_model.sav")):
        # Cumulative Displacement Curves
        print("---| Processing Rows |---")
        #cdc = Analysis().singlex(TrainDataset) # CDC for a single point (1,1)
        #dataset = Analysis().multiplex(TrainDataset) # CDC for all points in video (array of points containing array of coords throughout video)
        dataset, total_frames = Analysis().all_xy(TrainDataset) # CDC for all points in video (array of points containing array of coords throughout video)

        # ELAN File Import
        print("---| Importing ELAN File |---")
        frame_points, tiers = Analysis().elanImport(GroundTruth_Train)

        #print("---| TIERS |---")
        #print(tiers)

        #Visual for displacement points compared to mapped out frames
        if testing:
            print("---| Generating Visual of Training Data |---")
            visual_x = Analysis().multiplex(TrainDataset)
            for point in range(len(visual_x)):
                plt.plot(visual_x[point])
            for line in frame_points:
                plt.axvline(x=int(line), color='r')
            plt.show()

        print("---| SVM Binary Classifier |---")
        ground_truth = Analysis().SVMPreparer(tiers, total_frames)
        model = Analysis().SVMClassifier(dataset, ground_truth)
    else:
        print("---| Found premade model, loading in |---")
        model = pickle.load(open("trained_model.sav", 'rb'))

    timechanges, predictions = Analysis().testSVM(model, TestDataset, "", OutputFile)

    if testing:
        print("---| Generating Visual of Produced Data |---")
        testoptic = Analysis().multiplex(TestDataset)
        for point in range(len(testoptic)):
            plt.plot(testoptic[point])
        for line in timechanges:
            plt.axvline(x=int(line), color='r')
        plt.show()

    print('---| Pushing Results to ' + OutputFile + ' |---')
    Analysis().elanExport(timechanges, predictions, OutputFile) # Export to scrollable ELAN file

    print('---| Analysis Completed |---')
    print("Time to complete: " + str(datetime.now() - startTime))

if __name__ == '__main__':
    main()