import os
import shutil
import sys
import numpy
import cv2
import pandas
import math
import yaml
from PyQt6 import QtCore
from PyQt6.QtCore import Qt,QPoint
from PyQt6.QtWidgets import QProgressBar,QApplication,QDoubleSpinBox,QLabel,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QPushButton,QSpacerItem,QFileDialog,QTabWidget,QComboBox,QCheckBox,QSlider,QMainWindow,QLineEdit
from PyQt6.QtGui import QImage,QPixmap,QShortcut,QKeySequence
from superqt import QRangeSlider

class ImageLabel(QLabel):
    def __init__(self,parent=None):
        super(QLabel,self).__init__(parent)
        self.setMouseTracking(False)


class CREATE_Challenge_Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.imgShape = (480, 640, 3)
        self.originalImageShape = None
        self.setMouseTracking(False)
        self.modifyBBoxStarted = False
        self.yolo = None
        self.datasetDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Datasets")
        if not os.path.exists(self.datasetDir):
            os.mkdir(self.datasetDir)
        self.datacsv = None
        self.imageDirectory = None
        self.dataset = None
        self.displayMode = "show all"
        self.allDisplayModes = ["show all", "show active", "hide all"]
        self.setWindowTitle("CREATE Challenge Annotator")
        self.setupWidget()
        self.setupWidgetConnections()
        self.show()

    def setupWidget(self):
        titleMsg = QLabel("<h1>CREATE Challenge Annotator<h1>")
        titleMsg.move(20, 20)
        mainLayout = QHBoxLayout()

        #############################
        # Create Image manipulations layout
        #############################
        layout = QVBoxLayout()
        layout.addWidget(titleMsg)
        hbox = QHBoxLayout()
        self.selectDatasetLabel = QLabel("Select Dataset:")
        self.selectDatasetComboBox = QComboBox()
        datasets = [x for x in os.listdir(self.datasetDir) if os.path.isdir(os.path.join(self.datasetDir,x))]
        self.selectDatasetComboBox.addItems(["Select dataset", "Create new dataset"] + datasets)
        hbox.addWidget(self.selectDatasetLabel)
        hbox.addWidget(self.selectDatasetComboBox)
        layout.addLayout(hbox)
        self.selectImageDirButton = QPushButton("Select Image Directory")
        self.selectImageDirButton.setEnabled(False)
        layout.addWidget(self.selectImageDirButton)
        self.imageDirectoryLabel = QLabel("Image Directory: ")
        layout.addWidget(self.imageDirectoryLabel)
        layout.addItem(QSpacerItem(100, 20))

        self.deleteBoxButton = QPushButton("Delete current box")
        layout.addWidget(self.deleteBoxButton)

        self.detectionWidget = QWidget()
        self.detectionLayout = QGridLayout()
        self.currentBoxSelectorLabel = QLabel("Current box")
        self.detectionLayout.addWidget(self.currentBoxSelectorLabel, 1, 0)
        self.currentBoxSelector = QComboBox()
        self.detectionLayout.addWidget(self.currentBoxSelector, 1, 1,1,2)

        self.classSelectorLabel = QLabel("Class name")
        self.detectionLayout.addWidget(self.classSelectorLabel, 2, 0)
        self.classSelector = QComboBox()
        self.classSelector.addItems(["Select class name","Add new class"])
        self.detectionLayout.addWidget(self.classSelector, 2, 1, 1, 2)

        self.xCoordinateLabel = QLabel("X Coordinates")
        self.detectionLayout.addWidget(self.xCoordinateLabel, 3, 0)
        self.xminSelector = QDoubleSpinBox()
        self.xminSelector.setMaximum(1.0)
        self.xminSelector.setMinimum(0.0)
        self.xminSelector.setSingleStep(0.05)
        self.xmaxSelector = QDoubleSpinBox()
        self.xmaxSelector.setMaximum(1.0)
        self.xmaxSelector.setMinimum(0.0)
        self.xmaxSelector.setSingleStep(0.05)
        self.detectionLayout.addWidget(self.xminSelector, 3, 1)
        self.detectionLayout.addWidget(self.xmaxSelector, 3, 2)

        self.yCoordinateLabel = QLabel("Y Coordinates")
        self.detectionLayout.addWidget(self.yCoordinateLabel, 4, 0)
        self.yminSelector = QDoubleSpinBox()
        self.yminSelector.setMaximum(1.0)
        self.yminSelector.setMinimum(0.0)
        self.yminSelector.setSingleStep(0.05)
        self.ymaxSelector = QDoubleSpinBox()
        self.ymaxSelector.setMaximum(1.0)
        self.ymaxSelector.setMinimum(0.0)
        self.ymaxSelector.setSingleStep(0.05)
        self.detectionLayout.addWidget(self.yminSelector, 4, 1)
        self.detectionLayout.addWidget(self.ymaxSelector, 4, 2)
        self.detectionWidget.setLayout(self.detectionLayout)
        layout.addWidget(self.detectionWidget)

        mainLayout.addLayout(layout)
        mainLayout.addItem(QSpacerItem(20, 100))

        #############################
        # Create Image layout
        #############################
        imageLayout = QVBoxLayout()
        self.imageLabel = ImageLabel()
        image = numpy.zeros(self.imgShape)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixelmap = QPixmap.fromImage(qImage)
        self.imageLabel.setPixmap(pixelmap)
        imageLayout.addWidget(self.imageLabel)
        mainLayout.addLayout(imageLayout)

        #############################
        # Create Classification layout
        #############################

        self.classificationWidget = QWidget()
        self.classificationLayout = QGridLayout()

        self.taskSelectorLabel = QLabel("Task name")
        self.classificationLayout.addWidget(self.taskSelectorLabel, 2, 0)
        self.taskSelector = QComboBox()
        self.taskSelector.addItems(["Select task name", "Add new task"])
        self.classificationLayout.addWidget(self.taskSelector, 2, 1, 1, 2)
        self.classificationWidget.setLayout(self.classificationLayout)
        layout.addWidget(self.classificationWidget)
        self.imageSeekWidget = QSlider(Qt.Orientation.Horizontal)
        self.imageSeekWidget.enabled = False
        layout.addWidget(self.imageSeekWidget)
        self.messageLabel = QLabel("")
        layout.addItem(QSpacerItem(100, 20))
        layout.addWidget(self.messageLabel)

        self.createDatasetWidget()
        self.createClassWidget()
        self.createTaskWidget()

        self.setLayout(mainLayout)
        self.setupHotkeys()

    def setupWidgetConnections(self):
        self.selectDatasetComboBox.currentIndexChanged.connect(self.onDatasetSelected)
        self.selectImageDirButton.clicked.connect(self.onSelectImageDirectory)
        self.currentBoxSelector.currentIndexChanged.connect(self.onCurrentBoxChanged)
        self.classSelector.currentIndexChanged.connect(self.onClassSelected)
        self.taskSelector.currentIndexChanged.connect(self.onTaskSelected)
        self.xminSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.xmaxSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.yminSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.ymaxSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.deleteBoxButton.clicked.connect(self.onDeleteBox)
        self.imageSeekWidget.valueChanged.connect(self.updateWidgetFromSeek)

    def mouseMoveEvent(self,event):
        if self.currentBoxSelector.currentIndex() >=2 and self.modifyBBoxStarted:
            cursorPosition = event.pos()
            cursorPosition = (cursorPosition.x(),cursorPosition.y())
            imageWidgetPosition = (self.imageLabel.x(),self.imageLabel.y())
            imageXCoordinate = max(0,min(self.imgShape[1],cursorPosition[0]-imageWidgetPosition[0]))
            imageYCoordinate = max(0,min(self.imgShape[0],cursorPosition[1]-imageWidgetPosition[1]))
            box_index = self.currentBoxSelector.currentIndex() - 2
            self.currentBBoxes = eval(str(self.currentBBoxes))
            self.currentBBoxes[box_index]["xmin"] = min(imageXCoordinate, self.startingPoint[0])/self.imgShape[1]
            self.currentBBoxes[box_index]["ymin"] = min(imageYCoordinate, self.startingPoint[1])/self.imgShape[0]
            self.currentBBoxes[box_index]["xmax"] = max(imageXCoordinate, self.startingPoint[0])/self.imgShape[1]
            self.currentBBoxes[box_index]["ymax"] = max(imageYCoordinate, self.startingPoint[1])/self.imgShape[0]
            #print(self.currentBBoxes)
            self.setBBoxCoordinates(self.currentBBoxes[box_index])
            self.setImage(self.currentImage, reload_image=False)

    def mousePressEvent(self,event):
        if self.currentBoxSelector.currentIndex() >=2:
            cursorPosition = event.pos()
            cursorPosition = (cursorPosition.x(), cursorPosition.y())
            imageWidgetPosition = (self.imageLabel.x(), self.imageLabel.y())
            imageWidgetShape = (self.imageLabel.width(),self.imageLabel.height())
            if imageWidgetPosition[0]<=cursorPosition[0]<=(imageWidgetPosition[0]+imageWidgetShape[0]) and imageWidgetPosition[1]<=cursorPosition[1]<=(imageWidgetPosition[1]+imageWidgetShape[1]):
                self.modifyBBoxStarted = True
                self.xminSelector.blockSignals(True)
                self.xmaxSelector.blockSignals(True)
                self.yminSelector.blockSignals(True)
                self.ymaxSelector.blockSignals(True)
                imageXCoordinate = max(0, min(self.imgShape[1], cursorPosition[0] - imageWidgetPosition[0]))
                imageYCoordinate = max(0, min(self.imgShape[0], cursorPosition[1] - imageWidgetPosition[1]))
                self.startingPoint = (imageXCoordinate,imageYCoordinate)
                box_index = self.currentBoxSelector.currentIndex() - 2
                self.currentBBoxes = eval(str(self.currentBBoxes))
                self.currentBBoxes[box_index]["xmin"] = imageXCoordinate/self.imgShape[1]
                self.currentBBoxes[box_index]["ymin"] = imageYCoordinate/self.imgShape[0]
                self.currentBBoxes[box_index]["xmax"] = imageXCoordinate/self.imgShape[1]
                self.currentBBoxes[box_index]["ymax"] = imageYCoordinate/self.imgShape[0]
                self.setBBoxCoordinates(self.currentBBoxes[box_index])
                self.setImage(self.currentImage,reload_image=False)
        else:
            self.modifyBBoxStarted = False

    def mouseReleaseEvent(self,event):
        if self.currentBoxSelector.currentIndex() >= 2 and self.modifyBBoxStarted:
            self.updateLabelFile()
            self.setImage(self.currentImage, reload_image=False)
            self.xminSelector.blockSignals(False)
            self.xmaxSelector.blockSignals(False)
            self.yminSelector.blockSignals(False)
            self.ymaxSelector.blockSignals(False)
        self.modifyBBoxStarted = False

    def setupHotkeys(self):
        allVerticalFlipShortcut = QShortcut(self)
        allVerticalFlipShortcut.setKey("v")
        allVerticalFlipShortcut.activated.connect(self.onFlipAllImageVClicked)

        allhorizontalFlipShortcut = QShortcut(self)
        allhorizontalFlipShortcut.setKey("h")
        allhorizontalFlipShortcut.activated.connect(self.onFlipAllImageHClicked)

        nextShortcut = QShortcut(self)
        nextShortcut.setKey("n")
        nextShortcut.activated.connect(self.showNextImage)

        previousShortcut = QShortcut(self)
        previousShortcut.setKey("p")
        previousShortcut.activated.connect(self.showPreviousImage)

        exportShortcut = QShortcut(self)
        exportShortcut.setKey("Ctrl+e")
        exportShortcut.activated.connect(self.ExportToLabelFile)

        displayModeShortcut = QShortcut(self)
        displayModeShortcut.setKey("m")
        displayModeShortcut.activated.connect(self.cycleDisplayMode)

    def cycleDisplayMode(self):
        currentMode = self.allDisplayModes.index(self.displayMode)
        newMode = (currentMode + 1)%len(self.allDisplayModes)
        self.displayMode = self.allDisplayModes[newMode]
        self.setImage(self.currentImage,reload_image=False)

    def ExportToLabelFile(self):
        completed_imgs = self.imageLabelFile.loc[
            (self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
        img_files = self.imageLabelFile.loc[self.imageLabelFile["Folder"] == self.imageDirectory]

        videoId = os.path.basename(os.path.dirname(self.imageDirectory))
        subtype = os.path.basename(self.imageDirectory)

        if os.path.exists(os.path.join(self.imageDirectory, "{}_{}_Labels.csv".format(videoId, subtype))):
            self.videoID = videoId
            self.subtype = subtype
            label_file_path = os.path.join(self.imageDirectory, "{}_{}_Labels.csv".format(videoId, subtype))
            self.labelFile = pandas.read_csv(label_file_path)

        elif os.path.exists(os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))):
            self.videoID = subtype
            self.subtype = None
            label_file_path = os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))
            self.labelFile = pandas.read_csv(label_file_path)

        else:
            self.videoID = subtype
            self.subtype = None
            label_file_path = os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))
            self.labelFile = pandas.DataFrame({"FileName": [img_files["FileName"][i] for i in img_files.index]})

        if len(completed_imgs.index) == len(img_files.index):
            idx_to_drop = []
            for i in self.labelFile.index:
                fileName = self.labelFile["FileName"][i]
                entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==fileName]
                if entry.empty:
                    idx_to_drop.append(i)

            self.labelFile = self.labelFile.drop(idx_to_drop,axis="index")
            self.labelFile.index = [i for i in range(len(self.labelFile.index))]

        print("exporting to label file: {}".format(label_file_path))

        self.labelFile["Tool bounding box"] = [self.convertBBoxes(img_files["Bounding boxes"][i]) if img_files["Status"][i]!="Review" else [] for i in img_files.index]
        self.labelFile["Overall Task"] = [img_files["Task"][i] if img_files["Status"][i]!="Review" else "nothing" for i in img_files.index]
        self.labelFile.to_csv(label_file_path,index=False)

    def convertBBoxes(self,bboxes):
        bboxes = eval(str(bboxes))
        for bbox in bboxes:
            bbox["xmin"] = int(bbox["xmin"]*self.originalImageShape[1])
            bbox["xmax"] = int(bbox["xmax"] * self.originalImageShape[1])
            bbox["ymin"] = int(bbox["ymin"] * self.originalImageShape[0])
            bbox["ymax"] = int(bbox["ymax"] * self.originalImageShape[0])
        return bboxes

    def normalizeBBoxes(self,filename,bboxes):
        img = cv2.imread(os.path.join(self.imageDirectory,filename))
        bboxes = eval(str(bboxes))
        for bbox in bboxes:
            bbox["xmin"] = float(bbox["xmin"])/img.shape[1]
            bbox["xmax"] = float(bbox["xmax"])/img.shape[1]
            bbox["ymin"] = float(bbox["ymin"])/img.shape[0]
            bbox["ymax"] = float(bbox["ymax"])/img.shape[0]
        return bboxes

    def getCurrentIndex(self,idx,indexes):
        idx_found = False
        i = -1
        while not idx_found and i<len(indexes)-1:
            i += 1
            if indexes[i] == idx:
                idx_found = True
        return i

    def checkForNoneBBoxes(self):
        bboxes = eval(str(self.currentBBoxes))
        found_None = False
        for box in bboxes:
            if box["class"] is None:
                found_None = True
        return found_None

    def updateWidgetFromSeek(self):
        val = self.imageSeekWidget.value()
        self.currentImage = self.imageFiles[val]
        entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        self.currentTask = self.imageLabelFile["Task"][entry.index[0]]
        self.currentBBoxes = self.imageLabelFile["Bounding boxes"][entry.index[0]]
        self.updateWidget()

    def showNextImage(self):
        images_to_review = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & ((self.imageLabelFile["Status"] == "Review") | (self.imageLabelFile["Status"] == "Reviewed"))]
        completed_imgs =  self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
        current_image = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        none_bboxes = self.checkForNoneBBoxes()
        if not none_bboxes:
            if not self.imageLabelFile["Status"][current_image.index[0]] == "Reviewed":
                self.imageLabelFile["Status"][current_image.index[0]] = "Reviewed"
            if not images_to_review.empty:
                img_idxs = images_to_review.index
                curr_idx = self.getCurrentIndex(current_image.index[0],img_idxs)
                if curr_idx < len(img_idxs)-1:
                    next_idx = img_idxs[curr_idx+1]
                    pos_increase = 1
                else:
                    next_idx = img_idxs[curr_idx]
                    pos_increase = 0
                    self.updateLabelFile()
                reviewed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]

            else:#if len(completed_imgs.index)==len(self.imageFiles):
                img_idxs = completed_imgs.index
                curr_idx = self.getCurrentIndex(current_image.index[0], img_idxs)
                if curr_idx < len(img_idxs)-1:
                    next_idx = img_idxs[curr_idx+1]
                    pos_increase = 1
                else:
                    next_idx = img_idxs[curr_idx]
                    pos_increase = 0
            self.currentImage = self.imageLabelFile["FileName"][next_idx]
            self.imageLabelFile["Bounding boxes"][next_idx] = self.imageLabelFile["Bounding boxes"][next_idx]
            if self.imageLabelFile["Status"][next_idx]=="Review":
                self.currentTask = self.taskSelector.currentText()
                self.updateLabelFile()
            else:
                self.currentTask = self.imageLabelFile["Task"][next_idx]
            self.currentBBoxes = self.imageLabelFile["Bounding boxes"][next_idx]
            seekBlocking = self.imageSeekWidget.blockSignals(True)
            currentidx = self.imageSeekWidget.value()
            self.imageSeekWidget.setValue(currentidx+pos_increase)
            self.imageSeekWidget.blockSignals(seekBlocking)
            self.updateWidget()
        else:
            self.messageLabel.setText("All bounding boxes must be assigned a class")

    def euclideanDistance(self,x1,y1,x2,y2):
        distance = ((x1-x2)**2 + (y1-y2)**2)**0.5
        return distance

    def calculateCornerDistance(self,bbox_1,bbox_2):
        distance = self.euclideanDistance(bbox_1["xmin"],bbox_1["ymin"],bbox_2["xmin"],bbox_2["ymin"])
        distance += self.euclideanDistance(bbox_1["xmin"],bbox_1["ymax"],bbox_2["xmin"],bbox_2["ymax"])
        distance += self.euclideanDistance(bbox_1["xmax"], bbox_1["ymin"], bbox_2["xmax"], bbox_2["ymin"])
        distance += self.euclideanDistance(bbox_1["xmax"], bbox_1["ymax"], bbox_2["xmax"], bbox_2["ymax"])
        return distance

    def findClosestBBox(self,target_bbox,bboxes):
        bestDistance = math.inf
        bestBox = None
        for box in bboxes:
            if box["class"] == target_bbox["class"]:
                distance =self.calculateCornerDistance(target_bbox,box)
                if distance < bestDistance:
                    bestDistance = distance
                    bestBox = box
        return bestBox

    def smoothBBoxes(self,bboxes,next_idx):
        imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory)]
        prev_idx = next_idx - 1
        try:
            prev_status = imgs["Status"][prev_idx]
            if prev_status != "Reviewed":
                prev_bboxes = eval(str(imgs["Bounding boxes"][prev_idx]))
                bboxes = eval(str(bboxes))
                bestBoxes = []
                for box in prev_bboxes:
                    closestBox = self.findClosestBBox(box,bboxes)
                    if closestBox!=None and not closestBox in bestBoxes:
                        bestBoxes.append(closestBox)
                    elif closestBox!=None and closestBox in bestBoxes:
                        bestBoxes.append(box)
                    elif closestBox is None and not box in bestBoxes:
                        bestBoxes.append(box)
                return bestBoxes
            else:
                return bboxes
        except:
            return bboxes



    def updateWidget(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        task_signals = self.taskSelector.blockSignals(True)
        self.taskSelector.setCurrentText(self.currentTask)

        self.updateBBoxSelector()
        if len(eval(str(self.currentBBoxes))) > 0:
            self.currentBoxSelector.setCurrentIndex(2)
        else:
            self.currentBoxSelector.setCurrentIndex(0)
        self.onCurrentBoxChanged()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.taskSelector.blockSignals(task_signals)
        self.setImage(self.currentImage)
        msg = ""
        imgs_remaining = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) &( self.imageLabelFile["Status"]=="Review")]
        reviewed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) &( self.imageLabelFile["Status"]=="Reviewed")]
        if len(imgs_remaining.index) > 0:
            msg += "{}/{} images remaining in current set\n".format(len(imgs_remaining.index),len(imgs_remaining.index) + len(reviewed_imgs.index))
        else:
            msg += "Video complete, please press Ctrl+e to export your labels.\n"
        msg += "{}/{} images annotated".format(len(reviewed_imgs.index),len(self.imageFiles))
        self.messageLabel.setText(msg)

    def showPreviousImage(self):
        images_to_review = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & ((self.imageLabelFile["Status"] == "Review") | (self.imageLabelFile["Status"] == "Reviewed"))]
        current_image = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        completed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
        if not images_to_review.empty:
            img_idxs = images_to_review.index
            curr_idx = self.getCurrentIndex(current_image.index[0],img_idxs)
            if curr_idx > 0:
                next_idx = img_idxs[curr_idx-1]
                pos_decrease = 1
            else:
                next_idx = img_idxs[curr_idx]
                pos_decrease = 0
            reviewed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
        elif len(completed_imgs.index) == len(self.imageFiles):
            img_idxs = completed_imgs.index
            curr_idx = self.getCurrentIndex(current_image.index[0], img_idxs)
            if curr_idx > 0:
                next_idx = img_idxs[curr_idx - 1]
                pos_decrease = 1
            else:
                next_idx = img_idxs[curr_idx]
                pos_decrease = 0
        self.currentImage = self.imageLabelFile["FileName"][next_idx]
        self.currentBBoxes = self.imageLabelFile["Bounding boxes"][next_idx]
        self.currentTask = self.imageLabelFile["Task"][next_idx]
        seekBlocking = self.imageSeekWidget.blockSignals(True)
        currentidx = self.imageSeekWidget.value()
        self.imageSeekWidget.setValue(currentidx - pos_decrease)
        self.imageSeekWidget.blockSignals(seekBlocking)
        self.updateWidget()

    def onFlipAllImageHClicked(self):
        for img in self.imageFiles:
            imagePath = os.path.join(self.imageDirectory, img)
            image = cv2.imread(imagePath)
            image = cv2.flip(image, 1)
            cv2.imwrite(imagePath, image)
            entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"] == img]
            flipped_boxes = self.FlipLabelsHorizontally(eval(str(entry["Bounding boxes"][entry.index[0]])))
            self.imageLabelFile["Bounding boxes"][entry.index[0]] = flipped_boxes
            if img == self.currentImage:
                self.currentBBoxes = flipped_boxes
        self.imageLabelFile.to_csv(
            os.path.join(self.datasetDir, self.selectDatasetComboBox.currentText(), "image_labels.csv"), index=False)
        # self.setImage(self.currentImage)
        self.updateWidget()

    def onFlipAllImageVClicked(self):
        for img in self.imageFiles:
            imagePath = os.path.join(self.imageDirectory, img)
            image = cv2.imread(imagePath)
            image = cv2.flip(image, 0)
            cv2.imwrite(imagePath,image)
            entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]== img]
            flipped_boxes = self.FlipLabelsVertically(eval(str(entry["Bounding boxes"][entry.index[0]])))
            self.imageLabelFile["Bounding boxes"][entry.index[0]] = flipped_boxes
            if img == self.currentImage:
                self.currentBBoxes = flipped_boxes
        self.imageLabelFile.to_csv(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(),"image_labels.csv"),index=False)
        #self.setImage(self.currentImage)
        self.updateWidget()

    def FlipLabelsVertically(self,bboxes):
        for bbox in bboxes:
            oldYmin = float(bbox["ymin"])
            oldYmax = float(bbox["ymax"])
            bbox["ymin"] = 1.0-oldYmax
            bbox["ymax"] = 1.0-oldYmin
            bbox["xmin"] = float(bbox["xmin"])
            bbox["xmax"] = float(bbox["xmax"])
        return bboxes

    def FlipLabelsHorizontally(self,bboxes):
        for bbox in bboxes:
            oldXmin = float(bbox["xmin"])
            oldXmax = float(bbox["xmax"])
            bbox["xmin"] = 1.0 - oldXmax
            bbox["xmax"] = 1.0 - oldXmin
            bbox["ymin"] = float(bbox["ymin"])
            bbox["ymax"] = float(bbox["ymax"])
        return bboxes

    def createDatasetWidget(self):
        self.datasetwindow = QWidget()
        self.datasetwindow.setWindowTitle("Create new dataset")
        layout = QVBoxLayout()
        create_dataset_name_label = QLabel("Dataset name:")
        self.create_dataset_line_edit = QLineEdit()
        accept_reject_layout = QHBoxLayout()
        self.accept_button = QPushButton("Ok")
        self.reject_button = QPushButton("Cancel")
        accept_reject_layout.addWidget(self.accept_button)
        accept_reject_layout.addWidget(self.reject_button)
        layout.addWidget(create_dataset_name_label)
        layout.addWidget(self.create_dataset_line_edit)
        layout.addLayout(accept_reject_layout)
        self.datasetwindow.setLayout(layout)

        self.accept_button.clicked.connect(self.onDatasetCreated)
        self.reject_button.clicked.connect(self.onDatasetCreationCancelled)

    def createTaskWidget(self):
        self.taskwindow = QWidget()
        self.taskwindow.setWindowTitle("Create new task")
        layout = QVBoxLayout()
        create_task_name_label = QLabel("Task name:")
        self.create_task_line_edit = QLineEdit()
        accept_reject_layout = QHBoxLayout()
        self.task_accept_button = QPushButton("Ok")
        self.task_reject_button = QPushButton("Cancel")
        accept_reject_layout.addWidget(self.task_accept_button)
        accept_reject_layout.addWidget(self.task_reject_button)
        layout.addWidget(create_task_name_label)
        layout.addWidget(self.create_task_line_edit)
        layout.addLayout(accept_reject_layout)
        self.taskwindow.setLayout(layout)

        self.task_accept_button.clicked.connect(self.onTaskAdded)
        self.task_reject_button.clicked.connect(self.onTaskAddedCancelled)

    def onTaskSelected(self):
        task_signals = self.taskSelector.blockSignals(True)
        if self.taskSelector.currentText() == "Add new task":
            self.taskwindow.show()
        elif self.taskSelector.currentText() != "Select task name":
            prev_task = self.currentTask
            self.currentTask = self.taskSelector.currentText()
            entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"] == self.currentImage]
            currentIndex = entry.index[0]
            extend = True
            if currentIndex>0 and self.imageLabelFile["Folder"][currentIndex-1]==self.imageLabelFile["Folder"][currentIndex]:
                if self.currentTask == self.imageLabelFile["Task"][currentIndex-1]:
                    extend = False
            elif currentIndex<len(self.imageLabelFile.index)-1 and self.imageLabelFile["Folder"][currentIndex+1]==self.imageLabelFile["Folder"][currentIndex]:
                if self.currentTask == self.imageLabelFile["Task"][currentIndex +1]:
                    extend=False
            if extend:
                entries = self.imageLabelFile.loc[self.imageLabelFile["Folder"] == self.imageLabelFile["Folder"][currentIndex]]
                for i in range(currentIndex,entries.index[-1]):
                    if self.imageLabelFile["Task"][i]==prev_task and self.imageLabelFile["Status"][i]!="Review":
                        self.imageLabelFile["Task"][i] = self.currentTask
                    else:
                        break
            self.updateLabelFile()
        self.taskSelector.blockSignals(task_signals)

    def createClassWidget(self):
        self.classwindow = QWidget()
        self.classwindow.setWindowTitle("Create new class")
        layout = QVBoxLayout()
        create_class_name_label = QLabel("Class name:")
        self.create_class_line_edit = QLineEdit()
        accept_reject_layout = QHBoxLayout()
        self.class_accept_button = QPushButton("Ok")
        self.class_reject_button = QPushButton("Cancel")
        accept_reject_layout.addWidget(self.class_accept_button)
        accept_reject_layout.addWidget(self.class_reject_button)
        layout.addWidget(create_class_name_label)
        layout.addWidget(self.create_class_line_edit)
        layout.addLayout(accept_reject_layout)
        self.classwindow.setLayout(layout)

        self.class_accept_button.clicked.connect(self.onClassAdded)
        self.class_reject_button.clicked.connect(self.onClassAddedCancelled)

    def onClassSelected(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.classSelector.currentText() == "Add new class":
            self.classwindow.show()
            #self.updateBBoxSelector()
        elif self.classSelector.currentText() != "Select class name":
            self.updateBBoxClass()
            self.updateBBoxSelector()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage,reload_image=False)

    def updateTask(self):
        self.currentTask = self.taskSelector.currentText()
        self.updateLabelFile()

    def updateBBoxClass(self):
        self.currentBBoxes = eval(str(self.currentBBoxes))
        className = self.classSelector.currentText()
        bbox_index = self.currentBoxSelector.currentIndex()-2
        try:
            self.currentBBoxes[bbox_index]["class"] = className
            self.updateLabelFile()
        except IndexError:
            pass

    def updateBBoxCoordinates(self):
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_index = self.currentBoxSelector.currentIndex()-2
        xmin = self.xminSelector.value()
        ymin = self.yminSelector.value()
        xmax = self.xmaxSelector.value()
        ymax = self.ymaxSelector.value()
        self.currentBBoxes[bbox_index]["xmin"] = xmin
        self.currentBBoxes[bbox_index]["xmax"] = xmax
        self.currentBBoxes[bbox_index]["ymin"] = ymin
        self.currentBBoxes[bbox_index]["ymax"] = ymax
        self.setImage(self.currentImage,reload_image=False)
        self.updateLabelFile()

    def onCurrentBoxChanged(self):
        box_index = self.currentBoxSelector.currentIndex()
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.currentBoxSelector.currentText() == "Add new box":
            bbox = {"class":None,"xmin":0.45,"ymin":0.45,"xmax":0.55,"ymax":0.55}
            self.setBBoxCoordinates(bbox)
            self.currentBBoxes.append(bbox)
            self.classSelector.setCurrentIndex(0)
            self.updateBBoxSelector()
            self.currentBoxSelector.setCurrentIndex(self.currentBoxSelector.count()-1)
        elif self.currentBoxSelector.currentText() != "Select box":
            bbox = self.currentBBoxes[box_index-2]
            self.setBBoxCoordinates(bbox)
            self.classSelector.setCurrentText(bbox["class"])
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage,reload_image=False)

    def onDeleteBox(self):
        box_index = self.currentBoxSelector.currentIndex()-2
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.currentBoxSelector!="Select box":
            try:
                self.currentBBoxes.pop(box_index)
                self.updateBBoxSelector()
                self.updateLabelFile()
            except IndexError:
                pass
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage,reload_image=False)


    def setBBoxCoordinates(self,bbox):
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        self.xminSelector.setValue(bbox["xmin"])
        self.xmaxSelector.setValue(bbox["xmax"])
        self.yminSelector.setValue(bbox["ymin"])
        self.ymaxSelector.setValue(bbox["ymax"])
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)

    def onDatasetCreated(self):
        dataset_signals = self.selectDatasetComboBox.blockSignals(True)
        self.datasetwindow.hide()
        datasetName = self.create_dataset_line_edit.text()
        if not os.path.exists(os.path.join(self.datasetDir,datasetName)):
            os.mkdir(os.path.join(self.datasetDir,datasetName))
            self.imageLabelFile = pandas.DataFrame(columns=["Folder","FileName","Status","Bounding boxes","Task"])
            self.imageLabelFile.to_csv(os.path.join(self.datasetDir,datasetName,"image_labels.csv"),index=False)
            self.selectDatasetComboBox.addItem(datasetName)
            self.selectDatasetComboBox.setCurrentText(datasetName)
            self.messageLabel.setText("")
            self.selectImageDirButton.setEnabled(True)
        else:
            self.messageLabel.setText("A dataset named {} already exists".format(datasetName))
            self.datasetwindow.show()
        self.selectDatasetComboBox.blockSignals(dataset_signals)

    def onDatasetCreationCancelled(self):
        self.selectDatasetComboBox.setCurrentIndex(0)
        self.datasetwindow.hide()

    def onTaskAdded(self):
        task_signals = self.taskSelector.blockSignals(True)
        self.taskwindow.hide()
        taskName = self.create_task_line_edit.text()
        task_names = sorted([self.taskSelector.itemText(i) for i in range(2,self.taskSelector.count()-1)])
        if not taskName in task_names:
            self.taskSelector.addItem(taskName)
            self.taskSelector.setCurrentText(taskName)
            self.updateTask()
        else:
            self.messageLabel.setText("A task named {} already exists".format(taskName))
            self.taskwindow.show()
        self.taskSelector.blockSignals(task_signals)
        self.setImage(self.currentImage,reload_image=False)

        if not os.path.exists(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(),"task_class_mapping.yaml")):
            task_mapping = dict(zip([i for i in range(len(task_names))],task_names))
            with open(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(), "task_class_mapping.yaml"), "w") as f:
                yaml.dump(task_mapping, f)

    def onTaskAddedCancelled(self):
        self.taskSelector.setCurrentIndex(0)
        self.taskwindow.hide()

    def onClassAdded(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        self.classwindow.hide()
        className = self.create_class_line_edit.text()
        class_names = sorted([self.classSelector.itemText(i) for i in range(2,self.classSelector.count()-1)])
        if not className in class_names:
            self.classSelector.addItem(className)
            self.classSelector.setCurrentText(className)
            self.updateBBoxClass()
            self.updateBBoxSelector()
        else:
            self.messageLabel.setText("A class named {} already exists".format(className))
            self.classwindow.show()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.setImage(self.currentImage,reload_image=False)

        if not os.path.exists(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(),"class_mapping.yaml")):
            class_mapping = dict(zip([i for i in range(len(class_names))],class_names))
            with open(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(), "class_mapping.yaml"), "w") as f:
                yaml.dump(class_mapping, f)

    def onClassAddedCancelled(self):
        self.classSelector.setCurrentIndex(0)
        self.classwindow.hide()

    def onDatasetSelected(self):
        if self.selectDatasetComboBox.currentText() == "Create new dataset":
            self.datasetwindow.show()
        elif self.selectDatasetComboBox.currentText() != "Select dataset":
            self.selectImageDirButton.setEnabled(True)
            datasetName = self.selectDatasetComboBox.currentText()
            self.imageLabelFile = pandas.read_csv(os.path.join(self.datasetDir,datasetName,"image_labels.csv"))
            if os.path.exists(os.path.join(self.datasetDir,datasetName,"class_mapping.yaml")):
                with open(os.path.join(self.datasetDir,datasetName,"class_mapping.yaml"),"r") as f:
                    class_mapping = yaml.safe_load(f)
                class_signals = self.classSelector.blockSignals(True)
                for key in class_mapping:
                    self.classSelector.addItem(class_mapping[key])
                self.classSelector.blockSignals(class_signals)
            if os.path.exists(os.path.join(self.datasetDir, datasetName, "task_class_mapping.yaml")):
                with open(os.path.join(self.datasetDir,datasetName,"task_class_mapping.yaml"),"r") as f:
                    task_class_mapping = yaml.safe_load(f)
                task_signals = self.taskSelector.blockSignals(True)
                for key in task_class_mapping:
                    self.taskSelector.addItem(task_class_mapping[key])
                self.taskSelector.blockSignals(task_signals)
        else:
            self.selectImageDirButton.setEnabled(False)


    def onSelectImageDirectory(self):
        window = QWidget()
        window.setWindowTitle("Select Image Directory")
        #try:
        self.imageDirectory = QFileDialog.getExistingDirectoryUrl(window,"C://").toString().replace("file:///","")

        self.imageDirectoryLabel.setText("Image Directory: \n{}".format(self.imageDirectory))

        label_filepath = os.path.join(self.datasetDir, self.selectDatasetComboBox.currentText(), "image_labels.csv")
        if os.path.exists(label_filepath):
            self.imageLabelFile = pandas.read_csv(label_filepath)
            imgs = self.imageLabelFile.loc[self.imageLabelFile["Folder"] == self.imageDirectory]
            if not imgs.empty:
                self.imageFiles = [self.imageLabelFile["FileName"][i] for i in imgs.index]
                self.all_bboxes = None
            else:
                self.getImageFileNames()
        else:
            self.getImageFileNames()
        if len(self.imageFiles) > 0:
            self.addImagesToLabelFile()
            seekBlocking = self.imageSeekWidget.blockSignals(True)
            self.imageSeekWidget.setMinimum(0)
            self.imageSeekWidget.setMaximum(len(self.imageFiles)-1)
            self.imageSeekWidget.enabled = True
            self.imageSeekWidget.blockSignals(seekBlocking)
        else:
            self.messageLabel.setText("No images found in directory")
            self.imageSeekWidget.enabled = False
            self.setImage()
        '''except:
            self.messageLabel.setText("No images found in directory")
            self.setImage()'''


    def getImageFileNames(self):
        videoId = os.path.basename(os.path.dirname(self.imageDirectory))
        subtype = os.path.basename(self.imageDirectory)
        if os.path.exists(os.path.join(self.imageDirectory,"{}_{}_Labels.csv".format(videoId,subtype))):
            self.videoID = videoId
            self.subtype = subtype
            self.labelFile = pandas.read_csv(os.path.join(self.imageDirectory,"{}_{}_Labels.csv".format(videoId,subtype)))
            self.imageFiles = [self.labelFile["FileName"][i] for i in self.labelFile.index]
            if "Tool bounding box" in self.labelFile:
                bboxes = [self.labelFile["Tool bounding box"][i] for i in self.labelFile.index]
                if len(set(bboxes)) == 1:
                    self.all_bboxes = None
                else:
                    self.all_bboxes = bboxes
            else:
                self.all_bboxes = None
            if "Overall Task" in self.labelFile:
                self.all_task_labels = [self.labelFile["Overall Task"][i] for i in self.labelFile.index]
                if len(list(set(self.all_task_labels)))==1:
                    self.all_task_labels = None
            else:
                self.all_task_labels = None

        elif os.path.exists(os.path.join(self.imageDirectory,"{}_Labels.csv".format(subtype))):
            self.videoID = subtype
            self.subtype = None
            self.labelFile = pandas.read_csv(os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype)))
            self.imageFiles = [self.labelFile["FileName"][i] for i in self.labelFile.index]
            if "Tool bounding box" in self.labelFile:
                self.all_bboxes = [self.labelFile["Tool bounding box"][i] for i in self.labelFile.index]
            else:
                self.all_bboxes = None
            if "Overall Task" in self.labelFile:
                self.all_task_labels = [self.labelFile["Overall Task"][i] for i in self.labelFile.index]
                if len(list(set(self.all_task_labels)))==1:
                    self.all_task_labels = None
            else:
                self.all_task_labels = None
        else:
            self.imageFiles = [x for x in os.listdir(self.imageDirectory) if (".jpg" in x) or (".png" in x)]
            self.all_bboxes = None
            self.all_task_labels = None

        if not self.all_task_labels is None:
            task_signals = self.taskSelector.blockSignals(True)
            current_Item_count = self.taskSelector.count()
            currentTasks = [self.taskSelector.itemText(i) for i in range(current_Item_count)]
            new_tasks = list(set(self.all_task_labels))
            for task in new_tasks:
                if not task in currentTasks:
                    self.taskSelector.addItem(task)
            self.taskSelector.blockSignals(task_signals)

        if not self.all_bboxes is None:
            class_signals = self.classSelector.blockSignals(True)
            current_Item_count = self.classSelector.count()
            currentClasses =  [self.classSelector.itemText(i) for i in range(current_Item_count)]
            new_classes = self.get_bbox_classes()
            for bbox_class in new_classes:
                if not bbox_class in currentClasses:
                    self.classSelector.addItem(bbox_class)
            self.classSelector.blockSignals(class_signals)


    def get_bbox_classes(self):
        classes = []
        for bbox_list in self.all_bboxes:
            bbox_list = eval(str(bbox_list))
            for bbox in bbox_list:
                if not bbox["class"] in classes:
                    classes.append(bbox["class"])
        return classes



    def addImagesToLabelFile(self):
        img_labels = self.imageLabelFile.loc[self.imageLabelFile["Folder"]==self.imageDirectory]
        message = ""
        if img_labels.empty:
            new_df = pandas.DataFrame({"Folder":[self.imageDirectory for i in self.imageFiles],
                                       "FileName": self.imageFiles,
                                       "Status":["Incomplete" for i in self.imageFiles],
                                       "Bounding boxes": [[] for i in self.imageFiles],
                                       "Task":["nothing" for i in self.imageFiles]})
            self.imageLabelFile = pandas.concat([self.imageLabelFile,new_df])
            self.imageLabelFile.index = [i for i in range(len(self.imageLabelFile.index))]
            datasetName = self.selectDatasetComboBox.currentText()
            message += self.selectInitialImages()
            self.imageLabelFile.to_csv(os.path.join(self.datasetDir, datasetName, "image_labels.csv"), index=False)

        first_image = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Review")]
        if not first_image.empty:
            message += "{} images remaining".format(len(first_image.index))

        else:
            first_image = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
            message += "Video complete. Please select the next video to annotate"

        self.messageLabel.setText(message)
        self.currentImage = first_image["FileName"][first_image.index[0]]
        self.currentBBoxes = first_image["Bounding boxes"][first_image.index[0]]
        self.currentTask = first_image["Task"][first_image.index[0]]
        seekBlocking = self.imageSeekWidget.blockSignals(True)
        currentidx = self.imageFiles.index(self.currentImage)
        self.imageSeekWidget.setValue(currentidx)
        self.imageSeekWidget.blockSignals(seekBlocking)
        self.updateWidget()

    def updateBBoxSelector(self):
        prev_index = self.currentBoxSelector.currentIndex()
        initial_count = self.currentBoxSelector.count()
        for i in range(self.currentBoxSelector.count() - 1, -1, -1):
            self.currentBoxSelector.removeItem(i)
        self.currentBoxSelector.addItem("Select box")
        self.currentBoxSelector.addItem("Add new box")
        bboxes = eval(str(self.currentBBoxes))
        boxNames = ["{}. {}".format(i+1,bboxes[i]["class"]) for i in range(len(bboxes))]
        if len(boxNames) > 0:
            self.currentBoxSelector.addItems(boxNames)
        if self.currentBoxSelector.count() < initial_count and self.currentBoxSelector.count()>2:
            self.currentBoxSelector.setCurrentIndex(prev_index-1)
        elif self.currentBoxSelector.count() == 2:
            self.currentBoxSelector.setCurrentIndex(0)
        else:
            self.currentBoxSelector.setCurrentIndex(prev_index)

    def updateLabelFile(self):
        entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        if not entry.empty:
            self.imageLabelFile["Bounding boxes"][entry.index[0]] = eval(str(self.currentBBoxes))
            self.imageLabelFile["Task"][entry.index[0]] = self.currentTask
            self.imageLabelFile.to_csv(os.path.join(self.datasetDir,self.selectDatasetComboBox.currentText(),"image_labels.csv"),index = False)
        class_names = sorted([self.classSelector.itemText(i) for i in range(2,self.classSelector.count())])
        class_mapping = dict(zip([i for i in range(len(class_names))], class_names))
        task_class_names = sorted([x for x in self.imageLabelFile["Task"].unique()])
        task_class_mapping = dict(zip([i for i in range(len(task_class_names))], task_class_names))
        with open(os.path.join(self.datasetDir, self.selectDatasetComboBox.currentText(), "class_mapping.yaml"), "w") as f:
            yaml.dump(class_mapping, f)
        with open(os.path.join(self.datasetDir, self.selectDatasetComboBox.currentText(), "task_class_mapping.yaml"), "w") as f:
            yaml.dump(task_class_mapping, f)

    def selectInitialImages(self):
        for fileName in self.imageFiles:
            entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==fileName]
            self.imageLabelFile["Status"][entry.index[0]] = "Review"
            if not (self.all_bboxes is None):
                idx = self.imageFiles.index(fileName)
                self.imageLabelFile["Bounding boxes"][entry.index[0]] = self.normalizeBBoxes(fileName,self.all_bboxes[idx])

            if not self.all_task_labels is None:
                idx = self.imageFiles.index(fileName)
                self.imageLabelFile["Task"][entry.index[0]] = self.all_task_labels[idx]
                self.imageLabelFile["Status"][entry.index[0]] = "Reviewed"
        return "added {} files to annotate".format(len(self.imageFiles))

    def setImage(self,fileName=None, reload_image=True):
        if fileName == None:
            img = numpy.zeros(self.imgShape)
        else:
            try:
                img = self.img.copy()
                if fileName!=self.prev_filename:
                    self.img = cv2.imread(os.path.join(self.imageDirectory, fileName))
                    self.originalImageShape = self.img.shape
                    self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                    self.img = cv2.resize(self.img, (self.imgShape[1], self.imgShape[0]), interpolation=cv2.INTER_AREA)
                    self.prev_filename = fileName
                    img = self.img.copy()
            except AttributeError:
                self.img = cv2.imread(os.path.join(self.imageDirectory, fileName))
                self.originalImageShape = self.img.shape
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                self.img = cv2.resize(self.img, (self.imgShape[1], self.imgShape[0]), interpolation=cv2.INTER_AREA)
                self.prev_filename = fileName
                img = self.img.copy()
            bboxes = eval(str(self.currentBBoxes))
            if self.displayMode != "hide all":
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    if i == self.currentBoxSelector.currentIndex()-2:
                        colour = (0,255,0)
                    else:
                        colour = (255,0,0)
                    if self.displayMode == "show all" or i == self.currentBoxSelector.currentIndex()-2:
                        img = cv2.rectangle(img, (int(bbox["xmin"]*self.imgShape[1]), int(bbox["ymin"]*self.imgShape[0])), (int(bbox["xmax"]*self.imgShape[1]), int(bbox["ymax"]*self.imgShape[0])), colour, 2)
                        img = cv2.putText(img, "{}. {}".format(i+1, bbox["class"]),
                                            (int(bbox["xmin"]*self.imgShape[1]), int(bbox["ymin"]*self.imgShape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2,
                                            cv2.LINE_AA)

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImage = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixelmap = QPixmap.fromImage(qImage)
        self.imageLabel.setPixmap(pixelmap)


if __name__ == "__main__":
    app = QApplication([])
    anReviewer = CREATE_Challenge_Annotator()
    sys.exit(app.exec())