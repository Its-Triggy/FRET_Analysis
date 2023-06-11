from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import *

import sys, time
import czifile
import numpy as np
import xml.etree.cElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import pandas as pd
import openpyxl

BG_COLOR = "gainsboro"
FG_COLOR = "black"
SELECTOR_COLOR = "red"

SELECTOR_WIDTH = 40
SELECTOR_HEIGHT = 40
PCTL = 99 # Which percentile pixel shows max value
		
# This class contains all relevant information about the CZI meta data, CZI image file, and various variations of the image 
class CZI(czifile.CziFile):
	#Initializes an object of class CZI. This runs once automatically upon instantiation of the object. 
	def __init__(self, file_name):
		czifile.CziFile.__init__(self, file_name)
		self.img = czifile.imread(file_name).squeeze() #Stores the original czi		
		
		if len(self.img.shape)==3:
			self.n_channels = self.img.shape[0] #Stores the number of channels in the czi file
			self.n_slices = 1 #Stores number of slices (time or z slice)
			self.width = self.img.shape[1] #Stores the width of the img in pixels
			self.height = self.img.shape[2] #Stores the height of the img in pixels
			
		if len(self.img.shape)==4:
			self.img_4D = czifile.imread(file_name).squeeze() #Stores the original czi	
			self.img = self.img_4D[:,0,:,:]
			self.current_slice = 0
			print(self.img_4D.shape)
			self.n_channels = self.img_4D.shape[0] #Stores the number of channels in the czi file
			self.n_slices = self.img_4D.shape[1] #Stores number of slices (time or z slice)
			self.width = self.img_4D.shape[2] #Stores the width of the img in pixels
			self.height = self.img_4D.shape[3] #Stores the height of the img in pixels
		
		self.mean_img = np.mean(self.img, axis=0) #Stores one img which is the mean of all channels (for display purposes)
		self.subtracted_img = self.get_subtracted_img(0) #Stores the original_img minus n=2 standard deviations
		self.mean_subtracted_img = np.mean(self.subtracted_img, axis=0) #Stores one img which is the mean of all subtracted_img channels
		
		#self.pi_img = self.get_pi_img(0) #pi = pixels of interest. Stores subtracted_img where pixel values are above the threshold, otherwise np.nan
		self.ranges = self.get_wavelength_ranges(self.metadata()) #Stores a list of wavelength ranges of the form (lower, upper) for each channel
		self.channel_assignment = ['--Unassigned--' for i in range(self.n_channels)]
	
	'''
	#Updates subtracted_img, mean_subtracted_img, pi_img, mean_pi_img based on n (number of std's to subtract off) and threshold, both supplied by user
	def update_imgs(self, n, thresh):
		self.subtracted_img = self.get_subtracted_img(n)
		self.mean_subtracted_img = np.mean(self.subtracted_img, axis=0)
		self.pi_img = self.get_pi_img(thresh)
		self.mean_pi_img = self.pi_img.mean(axis=0) 
	'''
	
	#Switches the selected slice for a 4D image
	def select_slice(self, slice, n_std):
		if slice >= self.n_slices:
			print(f"slice index too high. slice choice: {slice}, n_slices: {self.n_slices}")
			return None
		
		self.img = self.img_4D[:,slice,:,:]
		self.current_slice = slice
		
		self.mean_img = np.mean(self.img, axis=0) #Stores one img which is the mean of all channels (for display purposes)
		self.subtracted_img = self.get_subtracted_img(n_std) #Stores the original_img minus n=2 standard deviations
		self.mean_subtracted_img = np.mean(self.subtracted_img, axis=0) #Stores one img which is the mean of all subtracted_img channels
		
		
	#Returns the original_img minus n standard deviations. Values less than zero are replaced with zero. This is to remove noise.
	def get_subtracted_img(self, n):
		new = np.zeros(self.img.shape)
		std = self.img.std(axis=(1,2))
		for i in range(self.n_channels):
			new[i] = self.img[i]-n*std[i]
		new[new<0] = 0
		self.subtracted_img = new
		self.mean_subtracted_img = np.mean(self.subtracted_img, axis=0)
		return new 
	
	#pi = pixels of interest
	def get_pi_img(self, thresh):
		n = self.mean_subtracted_img.copy()
		n[self.mean_subtracted_img < thresh] = np.nan
		
		return n 
	
	def zscore(self, array_3d):
		means = np.mean(array_3d, axis=(1,2)).reshape(self.n_channels,1,1)
		stds = np.std(array_3d, axis=(1,2)).reshape(self.n_channels,1,1)
		return (array_3d-means)/stds
	'''
	#Returns 2 lists of length n_channel: mean pixel value for each image, and pixel standard deviation for each image
	def get_pi_stats(self):
		m = np.ma.masked_array(self.pi_img, np.isnan(self.pi_img)).mean(axis=(1,2))
		s = np.ma.masked_array(self.pi_img, np.isnan(self.pi_img)).std(axis=(1,2))
		return m, s 
	'''

	#Recursively prints all children of a branch of an XML. This is useful for reading the metadata file. 
	def printXML(self, root, n=0):
		for child in root:
			print ("\t"*n, child.tag, child.text)
			self.printXML(child, n+1) 
	
	#Recursively searched thru an XML and returns the first child with child.tag==tag. This is useful for reading the metadata file. 
	def searchXML(self, root, tag):
		r =  None
		for child in root:
			if (child.tag == tag):
				#printXML(child, 0)
				return child
			elif r == None:
				r = self.searchXML(child, tag)		
		return r 
	
	#Reads through the metadata file and returns a list of wavelength ranges of the form (lower, upper) for each channel
	def get_wavelength_ranges(self, meta):
		ranges = []
		root = ET.fromstring(meta)
		channels = self.searchXML(root, "Channels")
	
		if len(channels) != self.n_channels:
			print("Error identifying channel detection wavelength ranges.")
			return
		
		for channel in channels:
			dw = self.searchXML(channel, "Ranges")
			if dw is not None:
				lower = str(round(float(dw.text.split("-")[0]),2))
				upper = str(round(float(dw.text.split("-")[1]),2))
				ranges.append((lower, upper))
			else:
				dw = self.searchXML(channel, "EmissionWavelength")
				if dw is not None:
					ranges.append(("?", str(round(float(dw.text),2))))
				else:
				 	ranges.append(("?","?"))
				 
	
		return ranges 

# App container
class App(QMainWindow):

	def __init__(self):
		super().__init__()
		
		options = QFileDialog.Options()
		options = options or QFileDialog.DontUseNativeDialog
		self.file, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;CZI Files (*.czi)", options=options)
		
		if not self.file:
			self.close()
			exit()
						
		#Stores a CZI object which has all the data from the desired file
		self.czi = CZI(self.file) 
		
		#Setup 
		self.title = 'FRET Automated Analysis'
		self.left = 25
		self.top = 25
		self.width = 1200
		self.height = 800
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		# Tabs widget
		self.table_widget = TableWidget(self)
		self.setCentralWidget(self.table_widget)
		
		self.show()

# All analysis tools
class TableWidget(QWidget):
	
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)
		self.parent = parent

		self.czi = parent.czi
		self.img_width = self.czi.width
		self.img_height = self.czi.height
		
		self.layout = QVBoxLayout(self)
		
		# Initialize Menu
		self.init_menu()

		# Initialize tab screen
		self.tabs = QTabWidget()
		#self.tab_test = QWidget()
		self.tab2 = QWidget()
		self.tab3 = QWidget()
		self.tab4 = QWidget()
		
		# Add tabs
		#self.tabs.addTab(self.tab_test,"Test")
		self.tabs.addTab(self.tab2,"Explore")
		self.tabs.addTab(self.tab3,"Assign Channels")
		self.tabs.addTab(self.tab4,"Set Parameters")
		
		#self.init_tab_test_UI()
		self.init_tab2_UI()
		self.init_tab3_UI()
		self.init_tab4_UI()
		
		# Add tabs to widget
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)
		
		'''
		# Setup a timer to trigger updates
		self.timer = QtCore.QTimer()
		self.timer.setInterval(50)
		self.timer.timeout.connect(self.update_large_img)
		self.timer.start()
		'''
	
	def init_menu(self):
		# Menu bar
		main_menu=self.parent.menuBar()

		# File Menu
		file_menu=main_menu.addMenu('File')
		open_button=QtWidgets.QAction('Open', self.parent)
		open_button.setShortcut('Ctrl+O')
		open_button.setStatusTip('Open a new image')
		open_button.triggered.connect(self.open)
		file_menu.addAction(open_button)
		
		quit_button = QtWidgets.QAction('Quit', self.parent)
		quit_button.setShortcut('Ctrl+Q')
		quit_button.setStatusTip('Quit application')
		quit_button.triggered.connect(self.parent.close)
		file_menu.addAction(quit_button)
		
		# Export Menu
		export_menu = main_menu.addMenu('Export Data')
		export_img_button = QtWidgets.QAction('Export "best pixels" image', self.parent)
		export_img_button.setShortcut('Ctrl+P')
		export_img_button.setStatusTip('The "best pixles" image is the image on the bottom right of the "set parameters" tab')
		export_img_button.triggered.connect(self.export_bp_img)
		export_menu.addAction(export_img_button)

		export_data_button = QtWidgets.QAction('Export "best pixels" data', self.parent)
		export_data_button.setShortcut('Ctrl+S')
		export_data_button.setStatusTip('Export "best pixles" data along with channel values and analysis parameters')
		export_data_button.triggered.connect(self.export_bp_data)
		export_menu.addAction(export_data_button)

		
		# Settings Menu
		settings_menu = main_menu.addMenu('Settings')
		import_settings_button = QtWidgets.QAction('Import settings', self.parent)
		import_settings_button.setShortcut('Ctrl+I')
		import_settings_button.setStatusTip('Import application settings from a previous analysis')
		import_settings_button.triggered.connect(self.import_settings)
		settings_menu.addAction(import_settings_button)

		export_settings_button = QtWidgets.QAction('Export settings', self.parent)
		export_settings_button.setShortcut('Ctrl+E')
		export_settings_button.setStatusTip('Export current application settings')
		export_settings_button.triggered.connect(self.export_settings)
		settings_menu.addAction(export_settings_button)
	
	def open(self):
		self.parent.close()
		self.parent.__init__()
			
	def export_bp_img(self):
		if not self.range_checkBox.isChecked():
			zscores = [self.zscore_spinBox.value()]
		else:
			zscores = np.linspace(self.lowerRange_spinBox.value(), self.upperRange_spinBox.value(), self.n_steps_spinBox.value())
		
		path, _ = QFileDialog.getSaveFileName(filter='.png')
		path = path.split(".png")[0]
		for count, zscore in enumerate(zscores):
			figure = Figure(figsize=(20, 20), dpi=120) 
			axes = figure.add_subplot(1, 1, 1)
			axes.axis('off')
			figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

			axes.cla()
			axes.axis('off')

			thresh = self.threshold_spinBox.value()
			axes.imshow(self.czi.get_pi_img(thresh), vmin=0, vmax=np.percentile(self.czi.mean_img, PCTL))
			bps = self.calc_BP(zscore)
			if bps is not None:
				axes.plot(bps[1], bps[0], 'r.', markersize=15)	
		
			fontsize = 30
			axes.text(0, int(.98*self.img_height),
							"\nthreshold: " + str(self.threshold_spinBox.value()) 
							+ "\nn best pixels: " + str(len(bps[0]))
							+ "\nzscore: " + str(zscore)
							+ "\n" + self.parent.file, 
							fontsize = fontsize)
			figure.savefig(path + str(count).zfill(3) + ".png")

	def export_bp_data(self):
		path, _ = QFileDialog.getSaveFileName(filter=".xlsx(*.xlsx)")
		if path is None:
			return None
			
		if not self.range_checkBox.isChecked():
			zscores = [self.zscore_spinBox.value()]
		else:
			zscores = np.linspace(self.lowerRange_spinBox.value(), self.upperRange_spinBox.value(), self.n_steps_spinBox.value())
		
		#complete_df = pd.DataFrame(columns=["Is best pixel", 'x_value', 'y_value', 'n_STDs_subtracted', 'intensity_threshold',
		#									'zscore threshold', 'donor channel zscores', 'acceptor channel zscores',
		#									'donor_channels', 'acceptor_channels', 'n_donor_requirement', 'n_acceptor_reqirement',
		#									'n_total_img_pixels', 'n_pixels_of_interest', 'n_best_pixels',] \
		#									+ ['channel_'+str(i+1).zfill(2)+'_raw' for i in range(self.czi.n_channels)] \
		#									+ ['channel_'+str(i+1).zfill(2)+'_filtered' for i in range(self.czi.n_channels)])
		
		
		for count, zscore in enumerate(zscores):
			bps = self.calc_BP(zscore)
			if bps is None:
				continue

			x = list(bps[0])
			y = list(bps[1]) 
			bp_count = len(x)

			# Get ZScores of BPs (Only consider pixels with mean value above thresh)
			mask = np.zeros(self.czi.subtracted_img.shape, dtype=bool)
			mask[:,:,:] = self.czi.mean_subtracted_img[np.newaxis,:,:] < self.threshold_spinBox.value()
			masked_subtracted = np.ma.masked_array(self.czi.subtracted_img, mask=mask)
			zscore_mat = self.czi.zscore(masked_subtracted)

			# Add a few non-BPs
			(c_, x_, y_) = np.where(~mask)
			c_non = []
			x_non = []
			y_non = []

			non_bp_count = 0
			for i in range(len(c_)):
				if non_bp_count >= bp_count:
					break
				elif x_[i] not in x and y_[i] not in y:
					non_bp_count += 1
					c_non.append(c_[i])
					x_non.append(x_[i])
					y_non.append(y_[i])

			donor_zscores = zscore_mat[:, x+x_non, y+y_non][self.donor_channels].T
			acceptor_zscores = zscore_mat[:, x+x_non, y+y_non][self.acceptor_channels].T

			channel_vals_raw = np.array([self.parent.czi.img[:, i, j] for i,j in zip(x+x_non,y+y_non)])
			channel_vals_subtracted = np.array([self.parent.czi.subtracted_img[:, i, j] for i,j in zip(x+x_non,y+y_non)])

			df=pd.DataFrame({})

			df["Is best pixel"] = [True for i in range(bp_count)] + [False for i in range(non_bp_count)]
			df['x_value'] = x + x_non
			df['y_value'] = y + y_non
			n_std = self.dropdown_std.currentText()
			df['n_STDs_subtracted'] = [n_std for i in range(bp_count + non_bp_count)]
			thresh = self.threshold_spinBox.value()
			df['intensity_threshold'] = [thresh for i in range(bp_count + non_bp_count)]
			zscore_thresh = self.zscore_spinBox.value()
			df['zscore threshold'] = [zscore_thresh for i in range(bp_count + non_bp_count)]
			df['donor channel zscores'] = [list(zscore) for zscore in donor_zscores]
			df['acceptor channel zscores'] = [list(zscore) for zscore in acceptor_zscores]
			dc = [d+1 for d in self.donor_channels]
			df['donor_channels'] = [dc for i in range(bp_count + non_bp_count)]
			ac = [a+1 for a in self.acceptor_channels]
			df['acceptor_channels'] = [ac for i in range(bp_count + non_bp_count)]
			df['n_donor_requirement'] = [self.dropdown_donor.currentText() for i in range(bp_count + non_bp_count)]
			df['n_acceptor_reqirement'] = [self.dropdown_acceptor.currentText() for i in range(bp_count + non_bp_count)]
			df['n_total_img_pixels'] = [self.czi.mean_img.shape[0] * self.czi.mean_img.shape[1] for i in range(bp_count + non_bp_count)]
			pi = n_bp = len(bps[0])
			n_pi = np.count_nonzero(self.czi.mean_subtracted_img >= self.threshold_spinBox.value())
			df['n_pixels_of_interest'] = [n_pi for i in range(bp_count +  + non_bp_count)]
			df['n_best_pixels'] = [bp_count for i in range(bp_count +  + non_bp_count)]

			for i in range(self.czi.n_channels):
				df['channel_'+str(i+1).zfill(2)+'_raw'] = channel_vals_raw[:,i]
			for i in range(self.czi.n_channels):
				df['channel_'+str(i+1).zfill(2)+'_filtered'] = channel_vals_subtracted[:,i]
				
			
			if count == 0:
				writer = pd.ExcelWriter(path, engine = 'openpyxl')
				df.to_excel(writer, sheet_name = 'zscore='+str(zscore))
				writer.save()
				writer.close()
			
			else:
				book = openpyxl.load_workbook(path)
				writer = pd.ExcelWriter(path, engine = 'openpyxl')
				writer.book = book

				df.to_excel(writer, sheet_name = 'zscore='+str(zscore))
				writer.save()
				writer.close()


			#complete_df.to_excel(path, engine='openpyxl') 
	'''
	def init_tab_test_UI(self):
		# Create first tab
		self.tab_test.layout = QtWidgets.QHBoxLayout(self)
		self.tab_test.setLayout(self.tab_test.layout)
		left_layout = QtWidgets.QGridLayout()
		self.tab_test.layout.addLayout(left_layout)
		right_layout = QtWidgets.QVBoxLayout()
		self.tab_test.layout.addLayout(right_layout)
		# Button text
		self.label = QLabel(self)
		self.label.setText("This is a button:")
		self.label.adjustSize()
		left_layout.addWidget(self.label, 0, 0)
		# Button
		self.pushButton1 = QPushButton("Click me!")
		self.pushButton1.clicked.connect(self.on_click)
		left_layout.addWidget(self.pushButton1, 0, 1)
		# Dropdown text
		self.label = QLabel(self)
		self.label.setText("This is a dropdown:")
		self.label.adjustSize()
		left_layout.addWidget(self.label, 1, 0)
		# Dropdown
		self.dropdown = QtWidgets.QComboBox(self)
		self.dropdown.addItem("Test1")
		self.dropdown.addItem("Test2")
		self.dropdown.addItem("Test3")
		self.dropdown.activated.connect(self.onDropdownUpdate)
		left_layout.addWidget(self.dropdown, 1, 1)
		#Radio button Text 
		self.label = QLabel(self)
		self.label.setText("These are radio buttons:")
		self.label.adjustSize()
		left_layout.addWidget(self.label, 2, 0)
		# Radio Buttons
		radio_layout =  QtWidgets.QVBoxLayout()
		left_layout.addLayout(radio_layout, 2, 1)
		self.radio = QtWidgets.QRadioButton(self)
		self.radio.setText("Option1")
		self.radio.setChecked(True)
		radio_layout.addWidget(self.radio)
		self.radio2 = QtWidgets.QRadioButton(self)
		self.radio2.setText("Option2")
		radio_layout.addWidget(self.radio2)
		# Sider text
		self.label = QLabel(self)
		self.label.setText("This is a slider:")
		self.label.adjustSize()
		left_layout.addWidget(self.label, 3, 0)
		# Slider
		self.slider = QSlider(Qt.Horizontal, self)
		self.slider.setGeometry(50, 280, 200, 50)
		self.slider.setMinimum(0)
		self.slider.setMaximum(20)
		self.slider.setTickPosition(QSlider.TicksBelow)
		self.slider.setTickInterval(2)
		self.slider.valueChanged.connect(lambda x: print(self.slider.value(), end='\r'))
		left_layout.addWidget(self.slider, 3, 1)
	
		# plot
		self.figure3 = Figure(figsize=(1.1, 2.2), dpi=300) #figsize=(2, 2), dpi=100)
		self.axes3 = self.figure3.add_subplot(1, 1, 1)
		#self.axes3.axis('off')
		self.figure3.subplots_adjust(left=0, right=1, top=1, bottom=0)
		self.canvas3 = FigureCanvasQTAgg(self.figure3)
		self.large_img = self.axes3.barh([0, 1, 2, 3], [0, 1, 2, 3], )#imshow(self.display_img)
		self.tab_test.layout.addWidget(self.canvas3)
	'''
	def init_tab2_UI(self):
		# Create second tab
		self.tab2.outer_layout = QtWidgets.QVBoxLayout(self)
		self.tab2.layout = QtWidgets.QHBoxLayout(self)
		self.tab2.outer_layout.addLayout(self.tab2.layout)
		self.tab2.setLayout(self.tab2.outer_layout)
		left_layout = QtWidgets.QVBoxLayout()
		self.tab2.layout.addLayout(left_layout)
		right_layout = QtWidgets.QVBoxLayout()
		self.tab2.layout.addLayout(right_layout)
		
		# File label
		self.file_label = QLabel(f"Current file: {self.parent.file}")
		self.tab2.outer_layout.addWidget(self.file_label)
		
		# Slider text
		self.label_nav = QLabel(self)
		self.label_nav.setText("Navigate with these sliders")
		self.label_nav.adjustSize()
		self.label_nav.setFont(QFont('Arial', 25))
		left_layout.addWidget(self.label_nav)

		# Large Slider X
		self.slider_large_x = QSlider(Qt.Horizontal, self)
		self.slider_large_x.setGeometry(50, 280, 200, 50)
		self.slider_large_x.setMinimum(0)
		self.slider_large_x.setMaximum(self.img_width-SELECTOR_WIDTH-1)
		self.slider_large_x.setTickPosition(QSlider.NoTicks)
		self.slider_large_x.setTickInterval(1)
		self.slider_large_x.valueChanged.connect(self.update_large_selector)
		self.slider_large_x.valueChanged.connect(self.update_small_img)
		self.slider_large_x.valueChanged.connect(self.update_plot)
		left_layout.addWidget(self.slider_large_x)
		
		# Large Slider Y
		self.slider_large_y = QSlider(Qt.Vertical, self)
		self.slider_large_y.setInvertedAppearance(True)
		self.slider_large_y.setGeometry(50, 280, 200, 50)
		self.slider_large_y.setMinimum(0)
		self.slider_large_y.setMaximum(self.img_height-SELECTOR_HEIGHT-1)
		self.slider_large_y.setTickPosition(QSlider.NoTicks)
		self.slider_large_y.setTickInterval(1)
		self.slider_large_y.valueChanged.connect(self.update_large_selector)
		self.slider_large_y.valueChanged.connect(self.update_small_img)
		self.slider_large_y.valueChanged.connect(self.update_plot)
		left_layout.addWidget(self.slider_large_y)
		
		# Small Slider X
		self.slider_small_x = QSlider(Qt.Horizontal, self)
		self.slider_small_x.setGeometry(50, 280, 200, 50)
		self.slider_small_x.setMinimum(0)
		self.slider_small_x.setMaximum(SELECTOR_WIDTH-1)
		self.slider_small_x.setTickPosition(QSlider.NoTicks)
		self.slider_small_x.setTickInterval(1)
		self.slider_small_x.valueChanged.connect(self.update_small_selector)
		self.slider_small_x.valueChanged.connect(self.update_plot)
		left_layout.addWidget(self.slider_small_x)
		
		# Small Slider Y
		self.slider_small_y = QSlider(Qt.Vertical, self)
		self.slider_small_y.setInvertedAppearance(True)
		self.slider_small_y.setGeometry(50, 280, 200, 50)
		self.slider_small_y.setMinimum(0)
		self.slider_small_y.setMaximum(SELECTOR_HEIGHT-1)
		self.slider_small_y.setTickPosition(QSlider.NoTicks)
		self.slider_small_y.setTickInterval(1)
		self.slider_small_y.valueChanged.connect(self.update_small_selector)
		self.slider_small_y.valueChanged.connect(self.update_plot)
		left_layout.addWidget(self.slider_small_y)
		
		if self.czi.n_slices > 1:
			# Dropdown text
			self.label_slice = QLabel(self)
			self.label_slice.setText("Select slice:")
			self.label_slice.adjustSize()
			left_layout.addWidget(self.label_slice)

			# Dropdown
			self.dropdown_slice = QtWidgets.QComboBox(self)
			for i in range(self.czi.n_slices):
				self.dropdown_slice.addItem(f"Slice {i+1}")
			self.dropdown_slice.activated.connect(lambda x=i: self.change_slice(x))
			left_layout.addWidget(self.dropdown_slice)
		
		# Large Image
		self.display_img = self.czi.mean_img
		self.figure1 = Figure(figsize=(1.1, 1.1), dpi=300) 
		self.axes1 = self.figure1.add_subplot(1, 1, 1)
		self.axes1.axis('off')
		self.figure1.subplots_adjust(left=0, right=1, top=1, bottom=0)
		self.canvas1 = FigureCanvasQTAgg(self.figure1)
		self.large_img = self.axes1.imshow(self.display_img, vmin=0, vmax=np.percentile(self.czi.mean_img, PCTL))
		right_layout.addWidget(self.canvas1)
		self.axes1.patches = []
		self.axes1.add_patch(Rectangle((self.slider_large_x.value(), self.slider_large_y.value()), 
						SELECTOR_WIDTH, SELECTOR_HEIGHT, color=SELECTOR_COLOR, fill=False))	
		self.update_large_img()
		
		# Small Image
		self.display_img = self.czi.mean_img
		self.figure2 = Figure(figsize=(1.1, 1.1), dpi=300) 
		self.axes2 = self.figure2.add_subplot(1, 1, 1)
		self.axes2.axis('off')
		self.figure2.subplots_adjust(left=0, right=1, top=1, bottom=0)
		self.canvas2 = FigureCanvasQTAgg(self.figure2)
		self.small_img = self.axes2.imshow(self.display_img[self.slider_large_y.value():self.slider_large_y.value()+SELECTOR_WIDTH, 
										self.slider_large_x.value():self.slider_large_x.value()+SELECTOR_HEIGHT],
										vmin=0, vmax=np.percentile(self.czi.mean_img, PCTL))
		right_layout.addWidget(self.canvas2)
		self.update_small_img()
		
		# Plot
		self.figure3 = Figure(figsize=(4, 3), dpi=120)
		self.axes3 = self.figure3.add_subplot(1, 1, 1)
		self.figure3.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
		self.canvas3 = FigureCanvasQTAgg(self.figure3)
		self.tab2.layout.addWidget(self.canvas3)
		self.update_plot()
		
	def init_tab3_UI(self):
		# Create third tab
		self.tab3.layout = QtWidgets.QVBoxLayout(self)
		self.tab3.setLayout(self.tab3.layout)
		
		self.channel_labels = []
		self.dropdowns = []
		layouts = []

		for i in range(self.czi.n_channels):
			layout = QtWidgets.QHBoxLayout()
			layouts.append(layout)
			self.tab3.layout.addLayout(layout)

			# Channel label
			label = QLabel(self)
			label.setText("Channel " + str(i+1) + ":")
			label.adjustSize()
			label.setFont(QFont('Arial', 20))
			layout.addWidget(label)
			self.channel_labels.append(label)

			# Dropdown
			dropdown = QtWidgets.QComboBox(self)
			dropdown.addItems(["--Unassigned--", "Donor", "I", "Acceptor"])
			dropdown.activated.connect(self.update_channel_assignment)
			layout.addWidget(dropdown)
			self.dropdowns.append(dropdown)

			# Channel range label
			label2 = QLabel(self)
			label2.setText(f"Wavelength range (nm): {self.czi.ranges[i][0]} - {self.czi.ranges[i][1]}")
			label2.adjustSize()
			label2.setFont(QFont('Arial', 10))
			label2.setAlignment(Qt.AlignCenter)
			layout.addWidget(label2)

			# Channel mean label
			label3 = QLabel(self)
			label3.setText(f"Mean intensity: {np.mean(self.czi.img[i]):.2f}")
			label3.adjustSize()
			label3.setFont(QFont('Arial', 10))
			label3.setAlignment(Qt.AlignCenter)
			layout.addWidget(label3)

			# Channel std label
			label4 = QLabel(self)
			label4.setText(f"STD intensity: {np.std(self.czi.img[i]):.2f}")
			label4.adjustSize()
			label4.setFont(QFont('Arial', 10))
			label4.setAlignment(Qt.AlignCenter)
			layout.addWidget(label4)

	def init_tab4_UI(self):
		self.n_donor = 0
		self.n_acceptor = 0
		self.donor_channels = []
		self.acceptor_channels = []

		# Create fourth tab
		self.tab4.layout = QtWidgets.QVBoxLayout(self)
		self.tab4.setLayout(self.tab4.layout)
		top_layout = QtWidgets.QHBoxLayout()
		self.tab4.layout.addLayout(top_layout)
		bottom_layout = QtWidgets.QHBoxLayout()
		self.tab4.layout.addLayout(bottom_layout)
		

		bottom_left_layout = QtWidgets.QVBoxLayout()
		bottom_layout.addLayout(bottom_left_layout)
		bottom_center_layout = QtWidgets.QVBoxLayout()
		bottom_layout.addLayout(bottom_center_layout)
		bottom_right_layout = QtWidgets.QVBoxLayout()
		bottom_layout.addLayout(bottom_right_layout)

		top_left_layout = QtWidgets.QVBoxLayout()
		top_layout.addLayout(top_left_layout)
		top_right_layout = QtWidgets.QVBoxLayout()
		top_layout.addLayout(top_right_layout)
		
		# Hist for threshold
		self.figure4 = Figure(figsize=(4, 5), dpi=120)
		self.axes4 = self.figure4.add_subplot(1, 1, 1)
		self.figure4.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
		self.canvas4 = FigureCanvasQTAgg(self.figure4)
		bottom_left_layout.addWidget(self.canvas4)

		# Threshold text
		threshold_layout = QtWidgets.QHBoxLayout()
		bottom_left_layout.addLayout(threshold_layout)
		self.label_thresh = QLabel(self)
		self.label_thresh.setText("Adjust intensity threshold: ")
		self.label_thresh.adjustSize()
		self.label_thresh.setFont(QFont('Arial', 20))
		threshold_layout.addWidget(self.label_thresh)
		
		#Threshold spinbox
		self.threshold_spinBox = QSpinBox()
		threshold_layout.addWidget(self.threshold_spinBox)
		self.threshold_spinBox.setSingleStep(10)
		self.threshold_spinBox.setMinimum(0)
		self.threshold_spinBox.setMaximum(np.max(self.czi.mean_img).astype(int))
		self.threshold_spinBox.valueChanged.connect(self.update_thresh_hist)
		self.threshold_spinBox.valueChanged.connect(self.update_zscore_hist)
		#self.threshold_spinBox.valueChanged.connect(self.update_thresh_label)
		self.threshold_spinBox.valueChanged.connect(self.update_BP_overlay_img)
		
		# Threshold slider
		'''
		self.slider_thresh = QSlider(Qt.Horizontal, self)
		self.slider_thresh.setGeometry(50, 280, 200, 50)
		self.slider_thresh.setMinimum(0)
		self.slider_thresh.setMaximum(np.max(self.czi.mean_img).astype(int))
		self.slider_thresh.setTickPosition(QSlider.TicksBelow)
		self.slider_thresh.setTickInterval(500)
		self.slider_thresh.valueChanged.connect(self.update_thresh_hist)
		self.slider_thresh.valueChanged.connect(self.update_zscore_hist)
		self.slider_thresh.valueChanged.connect(self.update_thresh_label)
		self.slider_thresh.valueChanged.connect(self.update_BP_overlay_img)
		bottom_left_layout.addWidget(self.slider_thresh)
		'''
		self.update_thresh_hist()

		# Hist for zscore
		self.figure5 = Figure(figsize=(4, 5), dpi=120)
		self.axes5 = self.figure5.add_subplot(1, 1, 1)
		self.figure5.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
		self.canvas5 = FigureCanvasQTAgg(self.figure5)
		bottom_center_layout.addWidget(self.canvas5)

		# Zscore text
		zscore_layout = QtWidgets.QHBoxLayout()
		bottom_center_layout.addLayout(zscore_layout)
		self.label_zscore = QLabel(self)
		self.label_zscore.setText("Adjust BP z-score requirement: ")
		self.label_zscore.adjustSize()
		self.label_zscore.setFont(QFont('Arial', 20))
		zscore_layout.addWidget(self.label_zscore)
		
		# Zscore spinbox
		self.zscore_spinBox = QDoubleSpinBox()
		zscore_layout.addWidget(self.zscore_spinBox)
		self.zscore_spinBox.setSingleStep(0.05)
		self.zscore_spinBox.setMinimum(0)
		self.zscore_spinBox.setMaximum(10)
		self.zscore_spinBox.valueChanged.connect(self.update_zscore_hist)
		#self.zscore_spinBox.valueChanged.connect(self.update_zscore_label)
		self.zscore_spinBox.valueChanged.connect(self.update_BP_overlay_img)
		
		# Zscore slider
		'''
		self.slider_zscore = QSlider(Qt.Horizontal, self)
		self.slider_zscore.setGeometry(50, 280, 200, 50)
		self.slider_zscore.setMinimum(0)
		self.slider_zscore.setMaximum(5*Z_FINE) # gets /Z_FINE, so max=5, but with high precision
		self.slider_zscore.setTickPosition(QSlider.TicksBelow)
		self.slider_zscore.setTickInterval(1)
		self.slider_zscore.valueChanged.connect(self.update_zscore_hist)
		self.slider_zscore.valueChanged.connect(self.update_zscore_label)
		self.slider_zscore.valueChanged.connect(self.update_BP_overlay_img)
		bottom_center_layout.addWidget(self.slider_zscore)
		'''
		self.update_zscore_hist()
		
		# Zscore range
		self.range_checkBox = QtWidgets.QCheckBox("Export range of z-scores")
		self.range_checkBox.stateChanged.connect(self.range_checkBox_changed)
		bottom_right_layout.addWidget(self.range_checkBox)
		range_layout = QtWidgets.QVBoxLayout()
		bottom_right_layout.addLayout(range_layout)
		self.lowerRange_spinBox = QDoubleSpinBox()
		self.lowerRange_spinBox.valueChanged.connect(self.lowerRange_spinBox_changed)
		self.lowerRange_spinBox.setSingleStep(0.05)
		self.lowerRange_spinBox.setMinimum(0)
		self.lowerRange_spinBox.setMaximum(10)
		self.upperRange_spinBox = QDoubleSpinBox()
		self.upperRange_spinBox.valueChanged.connect(self.upperRange_spinBox_changed)
		self.upperRange_spinBox.setSingleStep(0.05)
		self.upperRange_spinBox.setMinimum(0.01)
		self.upperRange_spinBox.setMaximum(10)
		self.n_steps_spinBox = QSpinBox()
		self.n_steps_spinBox.setMinimum(0)
		self.lowerRange_spinBox.setEnabled(False)
		self.upperRange_spinBox.setEnabled(False)
		self.n_steps_spinBox.setEnabled(False)
		self.lowerRange_label = QLabel("lower:")
		self.lowerRange_label.setEnabled(False)
		range_layout.addWidget(self.lowerRange_label)
		range_layout.addWidget(self.lowerRange_spinBox)
		self.upperRange_label = QLabel("upper:")
		self.upperRange_label.setEnabled(False)
		range_layout.addWidget(self.upperRange_label)
		range_layout.addWidget(self.upperRange_spinBox)
		self.n_steps_label = QLabel("n steps:")
		self.n_steps_label.setEnabled(False)
		range_layout.addWidget(self.n_steps_label)
		range_layout.addWidget(self.n_steps_spinBox)
		

		#BP percent label
		self.percent_BP_label = QLabel(self)
		self.percent_BP_label.setText("0% BP") # replaced with calc_bp
		self.percent_BP_label.adjustSize()
		self.percent_BP_label.setFont(QFont('Arial', 20))
		self.percent_BP_label.setAlignment(Qt.AlignCenter)
		top_right_layout.addWidget(self.percent_BP_label)

		# BP Overlay Image
		self.display_img = self.czi.get_pi_img(0)
		self.figure6 = Figure(figsize=(6, 6), dpi=120) 
		self.axes6 = self.figure6.add_subplot(1, 1, 1)
		self.axes6.axis('off')
		self.figure6.subplots_adjust(left=0, right=1, top=1, bottom=0)
		self.canvas6 = FigureCanvasQTAgg(self.figure6)
		top_right_layout.addWidget(self.canvas6)

		# subtract n STD label
		std_label = QLabel(self)
		std_label.setText("Remove noise; subtract n STDs:")
		std_label.adjustSize()
		std_label.setFont(QFont('Arial', 20))
		top_left_layout.addWidget(std_label)
		# Dropdown subtract n STD
		self.dropdown_std = QtWidgets.QComboBox(self)
		self.dropdown_std.addItems([str(i) for i in range(10)])
		self.dropdown_std.activated.connect(self.update_subtract_n_std)
		self.dropdown_std.activated.connect(self.update_BP_overlay_img)
		top_left_layout.addWidget(self.dropdown_std)

		# select n donor label
		donor_label = QLabel(self)
		donor_label.setText("How many donor channels must comply?")
		donor_label.adjustSize()
		donor_label.setFont(QFont('Arial', 20))
		top_left_layout.addWidget(donor_label)
		# Dropdown select n donor
		self.dropdown_donor = QtWidgets.QComboBox(self)
		self.dropdown_donor.addItems(["--No channels assigned--"])
		self.dropdown_donor.activated.connect(self.update_BP_overlay_img)
		self.dropdown_donor.setCurrentIndex(0)
		top_left_layout.addWidget(self.dropdown_donor)

		# select n acceptor label
		acceptor_label = QLabel(self)
		acceptor_label.setText("How many acceptor channels must comply?")
		acceptor_label.adjustSize()
		acceptor_label.setFont(QFont('Arial', 20))
		top_left_layout.addWidget(acceptor_label)
		# Dropdown select n acceptor
		self.dropdown_acceptor = QtWidgets.QComboBox(self)
		self.dropdown_acceptor.addItems(["--No channels assigned--"])
		self.dropdown_acceptor.activated.connect(self.update_BP_overlay_img)
		self.dropdown_acceptor.setCurrentIndex(0)
		top_left_layout.addWidget(self.dropdown_acceptor)
		
		if self.czi.n_slices > 1:	
			# select n slice label
			n_slice_label = QLabel(self)
			n_slice_label.setText("How many slices must agree?")
			n_slice_label.adjustSize()
			n_slice_label.setFont(QFont('Arial', 20))
			top_left_layout.addWidget(n_slice_label)
			# Dropdown select n slice
			self.dropdown_n_slice = QtWidgets.QComboBox(self)
			self.dropdown_n_slice.addItems(["Only the currently selected slice (See explore page)"])
			for i in range(self.czi.n_slices):
				self.dropdown_n_slice.addItems([str(i+1)])
			self.dropdown_n_slice.activated.connect(self.update_BP_overlay_img)
			self.dropdown_n_slice.setCurrentIndex(0)
			top_left_layout.addWidget(self.dropdown_n_slice)
		
		self.update_BP_overlay_img()
	
	def lowerRange_spinBox_changed(self, value):
		if self.upperRange_spinBox.value() - self.lowerRange_spinBox.value() <= 0.0001:
			self.upperRange_spinBox.setValue(self.lowerRange_spinBox.value() + 0.05)
	
	def upperRange_spinBox_changed(self, value):
		if self.upperRange_spinBox.value() - self.lowerRange_spinBox.value() <= 0.0001:
			self.lowerRange_spinBox.setValue(max(0, self.upperRange_spinBox.value() - 0.05))
		
	def range_checkBox_changed(self, isChecked):
		self.lowerRange_spinBox.setEnabled(isChecked)
		self.upperRange_spinBox.setEnabled(isChecked)
		self.n_steps_spinBox.setEnabled(isChecked)
		self.lowerRange_label.setEnabled(isChecked)
		self.upperRange_label.setEnabled(isChecked)
		self.n_steps_label.setEnabled(isChecked)
		
		
	def change_slice(self, slice):
		self.czi.select_slice(slice, int(self.dropdown_std.currentText()))
		self.update_large_img()
		self.update_small_img()
		self.update_plot()
		self.update_thresh_hist()
		self.update_zscore_hist()
		self.update_BP_overlay_img()
		self.update_subtract_n_std()
		
	def update_large_selector(self):
		self.axes1.patches = []
		self.axes1.add_patch(Rectangle((self.slider_large_x.value(), self.slider_large_y.value()), 
														   SELECTOR_WIDTH, SELECTOR_HEIGHT, color=SELECTOR_COLOR, fill=False))	
		self.canvas1.draw()
		
	def update_large_img(self):
		#self.axes1.cla()
		self.large_img.set_data(self.czi.mean_img)
		self.update_large_selector()
		#self.canvas1.draw() #Occurs in update_selector	

	def update_small_selector(self):
		self.axes2.patches = []
		self.axes2.add_patch(Rectangle((self.slider_small_x.value()-1, self.slider_small_y.value()-1), 2, 2, color=SELECTOR_COLOR, fill=False))
		self.canvas2.draw()
		
	def update_small_img(self):
		self.small_img.set_data(self.czi.mean_img[self.slider_large_y.value():self.slider_large_y.value()+SELECTOR_WIDTH, 
								self.slider_large_x.value():self.slider_large_x.value()+SELECTOR_HEIGHT])
		self.update_small_selector()
		#self.canvas2.draw() #Occurs in update_selector
					
	def update_plot(self):
		self.axes3.cla()  # Clear the canvas.
		self.axes3.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='x', alpha=0.7)
		self.axes3.set_xticks([i for i in range(0,50000, 10000)])
		self.axes3.set_yticks([i for i in range(self.czi.n_channels)])
		self.axes3.set_yticklabels(["CH"+str(i+1) for i in range(self.czi.n_channels)])
		self.axes3.set_xlabel('Intensity')
		self.axes3.set_ylabel('Channel')
		x = self.slider_large_x.value() + self.slider_small_x.value()
		y = self.slider_large_y.value() + self.slider_small_y.value()
		colors = []
		for a in self.czi.channel_assignment:
			if a == "Donor":
				colors.append("blue")
			elif a == "I":
				colors.append("green")
			elif a == "Acceptor":
				colors.append("red")
			else:
				colors.append("grey")
		self.plot = self.axes3.barh([i for i in range(self.czi.n_channels)], self.czi.img[:,y, x], color=colors)
		self.axes3.set_xlim(0,np.max(self.czi.img))
		
		#self.axes3.barh([0, 1, 2, 3], [random.random(), random.random(), random.random(), random.random()])
		# Trigger the canvas to update and redraw.
		self.canvas3.draw()
	
	def update_thresh_hist(self):
		self.axes4.cla()  # Clear the canvas.
		self.axes4.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)
		self.axes4.set_xlabel('Intensity')
		self.axes4.set_ylabel('Number of Occurences')
		thresh = self.threshold_spinBox.value()
		vals = self.czi.mean_subtracted_img.flatten()
		N, bins, patches = self.axes4.hist(vals, bins=200)
		self.axes4.plot([thresh, thresh], [0, np.max(N)], 'r-')
		#self.axes4.set_xlim(0,np.max(self.czi.img))
		
		self.canvas4.draw()
	
	def update_zscore_hist(self):
		self.axes5.cla()  # Clear the canvas.
		self.axes5.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)
		self.axes5.set_xlabel('Intensity')
		self.axes5.set_ylabel('Number of Occurences')
		thresh = self.threshold_spinBox.value()
		zscore = self.zscore_spinBox.value()
		vals = self.czi.mean_subtracted_img.flatten()
		vals = vals[np.where(vals>thresh)]
		
		vals = (vals - np.mean(vals)) / np.std(vals)
		pos = zscore #np.mean(vals) + zscore * np.std(vals)
		neg = -zscore #np.mean(vals) - zscore * np.std(vals)
		N, bins, patches = self.axes5.hist(vals, bins=200)
		self.axes5.plot([pos, pos], [0, np.max(N)], 'r-')
		self.axes5.plot([neg, neg], [0, np.max(N)], 'r-')
		
		# Trigger the canvas to update and redraw.
		self.canvas5.draw()
	
	'''
	def update_thresh_label(self):
		thresh = self.threshold_spinBox.value()
		self.label_thresh.setText(f"Adjust intensity threshold: {thresh}")

	def update_zscore_label(self):
		zscore = np.round(self.zscore_spinBox.value(), 4)
		self.label_zscore.setText(f"Adjust BP z-score requirement: {zscore}")
	'''
	
	def update_BP_overlay_img(self):
		self.axes6.cla()
		self.axes6.axis('off')

		thresh = self.threshold_spinBox.value()
		self.overlay_img = self.axes6.imshow(self.czi.get_pi_img(thresh), vmin=0, vmax=np.percentile(self.czi.mean_img, PCTL))
		
		bps = self.calc_BP(self.zscore_spinBox.value())
		if bps is not None:
			self.axes6.plot(bps[1], bps[0], 'r.', markersize=3)	
		
		self.canvas6.draw()
	
	def calc_BP(self, zscore):
		if self.czi.n_slices==1 or self.dropdown_n_slice.currentIndex()==0:
			bps = self.calc_current_slice_BP(zscore)		
		else:
			matrix = np.zeros((self.czi.n_slices, self.czi.width, self.czi.height)).astype(bool)
			current_slice_dummy = self.czi.current_slice
			for i in range(self.czi.n_slices):
				self.czi.select_slice(i, int(self.dropdown_std.currentText()))
				slice_bps = self.calc_current_slice_BP(zscore)
				matrix[i][slice_bps] = True
			self.czi.select_slice(current_slice_dummy, int(self.dropdown_std.currentText()))
			
			# create matrix_count where each pixel is number of slices with that pixel as BP
			matrix_count = np.count_nonzero(matrix, axis=0)
			
			# collect pixels that fit count criteria
			bps = np.where(matrix_count >= self.dropdown_n_slice.currentIndex())
		
		if bps is None:
			return None
			
		n_bp = len(bps[0])
		n_pi = np.count_nonzero(self.czi.mean_subtracted_img >= self.threshold_spinBox.value())
		self.percent_BP_label.setText(f"{n_bp}/{n_pi} = {round(n_bp/n_pi*100,4)}% BP")
		
		return bps

	def calc_current_slice_BP(self, zscore):
		if self.n_donor == 0 or self.n_acceptor == 0:
			print("n_donor or n_acceptor == 0!")
			return None 

		d_req = int(self.dropdown_donor.currentText())
		a_req = int(self.dropdown_acceptor.currentText())

		# Only consider pixels with mean value above thresh
		mask = np.zeros(self.czi.subtracted_img.shape, dtype=bool)
		mask[:,:,:] = self.czi.mean_subtracted_img[np.newaxis,:,:] < self.threshold_spinBox.value()
		masked_subtracted = np.ma.masked_array(self.czi.subtracted_img, mask=mask)
		# Convert subtracted channels to zscore_matrix
		zscore_matrix = self.czi.zscore(masked_subtracted)
	
		# create donor_count where each pixel is number of donor channels below -zscore for that pixel 
		donor_count = np.count_nonzero(zscore_matrix[self.donor_channels] < -zscore, axis=0)

		# create acceptor_img where each pixel is number of acceptor channels above zscore for that pixel
		acceptor_count = np.count_nonzero(zscore_matrix[self.acceptor_channels] > zscore, axis=0)

		# collect pixels that fit count criteria for both above imgs
		bps = np.where((donor_count >= d_req) & (acceptor_count >= a_req))
				
		return bps
		
		
	def update_channel_assignment(self):
		for i, dropdown in enumerate(self.dropdowns):
			self.czi.channel_assignment[i] = dropdown.currentText()
		
		self.update_plot()
		self.update_BP_overlay_img()

		self.n_donor = len([True for d in self.dropdowns if d.currentText()=="Donor"])
		self.donor_channels = [i for i,d in enumerate(self.dropdowns) if d.currentText()=="Donor"]
		self.dropdown_donor.clear()
		if self.n_donor == 0:
			self.dropdown_donor.addItems(["--No channels assigned--"])
			self.dropdown_donor.setCurrentIndex(0)
		else:
			self.dropdown_donor.addItems([str(i) for i in range(self.n_donor+1)])
			self.dropdown_donor.setCurrentIndex(self.n_donor)

		self.n_acceptor = len([True for d in self.dropdowns if d.currentText()=="Acceptor"])
		self.acceptor_channels = [i for i,d in enumerate(self.dropdowns) if d.currentText()=="Acceptor"]
		self.dropdown_acceptor.clear()
		if self.n_acceptor == 0:
			self.dropdown_acceptor.addItems(["--No channels assigned--"])
			self.dropdown_acceptor.setCurrentIndex(0)
		else:
			self.dropdown_acceptor.addItems([str(i) for i in range(self.n_acceptor+1)])
			self.dropdown_acceptor.setCurrentIndex(self.n_acceptor)

	def update_subtract_n_std(self):
		self.n_std = int(self.dropdown_std.currentText())
		self.czi.get_subtracted_img(self.n_std)
		self.update_BP_overlay_img()
		self.update_thresh_hist()
		self.update_zscore_hist()

	def export_settings(self):
		n_std = self.dropdown_std.currentText()
		thresh = self.threshold_spinBox.value()
		zscore = self.zscore_spinBox.value()
		d_list = self.donor_channels
		a_list = self.acceptor_channels
		donor_crit = self.dropdown_donor.currentIndex()
		acceptor_crit = self.dropdown_acceptor.currentIndex()

		df = pd.DataFrame({'n_STDs_subtracted':[n_std], 'pixel_threshold':[thresh], 'zscore_parameter':[zscore], 'donor_channels':[d_list], 'acceptor_channels':[a_list], 'donor_crit_index': [donor_crit], 'acceptor_crit_index': [acceptor_crit]})
		path, _ = QFileDialog.getSaveFileName(filter=".csv(*.csv)")
		if not path:
			return
		df.to_csv(path)

	def import_settings(self):
		options = QFileDialog.Options()
		options = options or QFileDialog.DontUseNativeDialog
		path, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;CSV Files (*.csv)", options=options)
		if not path:
			return
		df = pd.read_csv(path)

		self.dropdown_std.setCurrentText(str(df['n_STDs_subtracted'].values[0]))
		
		self.threshold_spinBox.setValue(int(df['pixel_threshold'].values[0]))
		self.zscore_spinBox.setValue(df['zscore_parameter'].values[0])
		d_list = eval(df['donor_channels'].values[0])
		a_list = eval(df['acceptor_channels'].values[0])
		
		for i, dropdown in enumerate(self.dropdowns):			
			if i in d_list:
				dropdown.setCurrentText('Donor')
			elif i in a_list:
				dropdown.setCurrentText('Acceptor')
			else:
				dropdown.setCurrentText('--Unassigned--')
		self.update_channel_assignment()

		print(int(df['donor_crit_index'].values[0]))
		self.dropdown_donor.setCurrentIndex(int(df['donor_crit_index'].values[0]))
		print(self.dropdown_donor.currentIndex())
		self.dropdown_acceptor.setCurrentIndex(int(df['acceptor_crit_index'].values[0]))

		self.update_subtract_n_std()
		self.update_thresh_hist()
		self.update_zscore_hist()
		self.update_BP_overlay_img()

def run():
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())
	
if __name__ == '__main__':
	run()
