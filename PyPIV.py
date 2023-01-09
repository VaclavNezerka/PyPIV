import os
import shutil
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QTextCursor
from screeninfo import get_monitors
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.registration import phase_cross_correlation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# get logo_image dimensions
# filepath_logo = "_aux/icon.png"
# logo_image = cv2.imread(filepath_logo)
# logo_height, logo_width = logo_image.shape[:2]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_boundaries(xy, box_size, n_cols, n_rows):
    min_x = int(max([0, xy[0]-np.round(box_size/2)]))
    max_x = int(min([xy[0]+np.round(box_size/2), n_cols-1]))
    min_y = int(max([0, xy[1]-np.round(box_size/2)]))
    max_y = int(min([xy[1]+np.round(box_size/2), n_rows-1]))
    return min_x, max_x, min_y, max_y


def get_scale(img_file, cur_scale):
    img = plt.imread(img_file)
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    [p1, p2] = plt.ginput(2, show_clicks=True)
    x = np.round([p1[0], p2[0]])
    y = np.round([p1[1], p2[1]])
    length = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
    def_length, ok_pressed = QInputDialog.getDouble(None,
                                                    "Get scale for the measurements", "Define a reference length:",
                                                    10, 0, 100000, 5)
    if ok_pressed:
        scale = length / def_length
    else:
        scale = cur_scale * 1.0
    plt.close(fig)
    return scale


def plot_deformed_grid(image_name, export_folder, xx, yy, subset_size, save_jpg):
    image = rgb2gray(mpimg.imread(image_name))
    # image_light = 255. * (image / 255.) ** 0.15
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_xlim(left=0, right=len(image[1]))
    ax.set_ylim(top=0, bottom=len(image))
    hs = 0.5 * subset_size
    if not isinstance(xx, list):  # if there is only one subset
        xx, yy = [xx], [yy]
    for i in range(len(xx)):
        ax.plot(xx[i], yy[i], '+', color='red', markersize=0.4, linewidth=0.4)
        x_coords = (xx[i] - hs, xx[i] + hs, xx[i] + hs, xx[i] - hs, xx[i] - hs)
        y_coords = (yy[i] - hs, yy[i] - hs, yy[i] + hs, yy[i] + hs, yy[i] - hs)
        ax.plot(x_coords, y_coords, '-', color='red', linewidth=0.4)
    plt.axis('off')
    plt.draw()
    if save_jpg:
        plt.savefig(export_folder + '.jpg', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(export_folder + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def plot_contours_irregular_grid(image_name, export_folder, xx, yy, z, fix_lims, fixed_lims, save_jpg):
    image = rgb2gray(mpimg.imread(image_name))
    # image_light = 255. * (image / 255.) ** 0.15
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_xlim(left=0, right=len(image[1]))
    ax.set_ylim(top=0, bottom=len(image))
    if fix_lims:
        ax.tricontourf(xx, yy, z, 20, cmap=cm.jet, vmin=fixed_lims[0], vmax=fixed_lims[1])
    else:
        ax.tricontourf(xx, yy, z, 20, cmap=cm.jet)  # the last parameter represents the number of contour levels
    ax.plot(xx, yy, 'ko', markersize=0.4)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(z)
    # colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(m, cax=cax, extend='both')
    plt.tight_layout()
    plt.draw()
    if save_jpg:
        plt.savefig(export_folder + '.jpg', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(export_folder + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def plot_quiver(image_name, export_folder, xx, yy, u, v, colors, arrow_width, arrow_length, fix_lims,
                fixed_lims, save_jpg):
    image = rgb2gray(mpimg.imread(image_name))
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_xlim(left=0, right=len(image[1]))
    ax.set_ylim(top=0, bottom=len(image))
    # arrow colors
    colormap = cm.jet
    m = cm.ScalarMappable(cmap=colormap)
    norm = Normalize()
    if fix_lims:
        norm.autoscale(fixed_lims)
        m.set_array(fixed_lims)
    else:
        norm.autoscale(colors)
        m.set_array(colors)
    # plot quiver
    ax.quiver(xx, yy, u, v, color=colormap(norm(colors)), angles='xy', units='width', pivot='mid',
              width=arrow_width, scale=1/arrow_length, scale_units='xy')
    # colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(m, cax=cax, extend='both')
    plt.draw()
    if save_jpg:
        plt.savefig(export_folder + '.jpg', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(export_folder + '.pdf', bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def create_video(file_names, video_file, video_duration=15):
    cv2_img = cv2.imread(file_names[0])
    height, width, layers = cv2_img.shape
    size = (width, height)

    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), len(file_names) / video_duration, size)
    for animation_image_name in file_names:
        cv2_img = cv2.imread(animation_image_name)
        out.write(cv2_img)
    out.release()


# get monitor resolution
for monitor in get_monitors():
    MainWindow_width = round(.7 * monitor.width)
    MainWindow_height = round(.7 * monitor.height)

canvas_maxWidth = round(0.82 * monitor.width) # ratio of the width of the area for buttons (343) and monitor width
                                                  # (1920) gives 0.82
canvas_maxHeight = round(1 * monitor.height)

# set global variables for multiple GUIs
subpixel_accuracy, frame_period = 0, 0
backward_analysis, update_subset_positions = True, True


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.main_window_handle = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(MainWindow_width, MainWindow_height)

        MainWindow.resizeEvent = self.on_window_resize

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, -1, 0, -1)
        self.verticalLayout.setObjectName("verticalLayout")

        self.radioButton_img = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_img.setChecked(True)
        self.radioButton_img.setObjectName("radioButton_img")
        self.verticalLayout.addWidget(self.radioButton_img)

        self.radioButton_mask = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_mask.setObjectName("radioButton_mask")
        self.verticalLayout.addWidget(self.radioButton_mask)

        self.radioButton_showGrid = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_showGrid.setObjectName("radioButton_showGrid")
        self.verticalLayout.addWidget(self.radioButton_showGrid)

        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)

        self.radioButton_quiver = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_quiver.setObjectName("radioButton_quiver")
        self.verticalLayout.addWidget(self.radioButton_quiver)

        self.radioButton_contour = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_contour.setObjectName("radioButton_contour")
        self.verticalLayout.addWidget(self.radioButton_contour)

        self.radioButton_subsets = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_subsets.setObjectName("radioButton_subsets")
        self.verticalLayout.addWidget(self.radioButton_subsets)

        self.gridLayout_3.addLayout(self.verticalLayout, 2, 6, 1, 1)

        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox.sizePolicy().hasHeightForWidth())
        self.spinBox.setSizePolicy(sizePolicy)
        self.spinBox.setMaximumSize(QtCore.QSize(60, 16777215))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(1)
        self.spinBox.setProperty("value", 1)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_3.addWidget(self.spinBox, 10, 1, 1, 1)

        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setEnabled(True)
        self.label_img.setMinimumSize(QtCore.QSize(400, 100))
        self.label_img.setMaximumSize(QtCore.QSize(canvas_maxWidth, canvas_maxHeight))
        self.label_img.setText("")
        self.label_img.setPixmap(QtGui.QPixmap("_aux/dummy_img.png").scaled(155, 97, QtCore.Qt.KeepAspectRatio))
        self.label_img.setScaledContents(True)
        self.label_img.setObjectName("label_img")
        self.gridLayout_3.addWidget(self.label_img, 0, 0, 10, 2)

        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout_3.addItem(spacerItem2, 3, 6, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem3, 2, 4, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(15, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem4, 2, 2, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(187, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem5, 3, 8, 1, 1)

        self.line = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.gridLayout_3.addWidget(self.line, 0, 3, 11, 1)

        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_3.addWidget(self.progressBar, 10, 6, 1, 3)

        self.label_fileName = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_fileName.sizePolicy().hasHeightForWidth())
        self.label_fileName.setSizePolicy(sizePolicy)
        self.label_fileName.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fileName.setObjectName("label_fileName")
        self.gridLayout_3.addWidget(self.label_fileName, 10, 0, 1, 1)

        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.gridLayout_3.addItem(spacerItem6, 1, 6, 1, 1)

        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setContentsMargins(0, -1, -1, -1)
        self.gridLayout_2.setObjectName("gridLayout_2")

        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem7, 0, 4, 1, 1)

        self.label_subsetSpacing = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_subsetSpacing.sizePolicy().hasHeightForWidth())
        self.label_subsetSpacing.setSizePolicy(sizePolicy)
        self.label_subsetSpacing.setObjectName("label_subsetSpacing")
        self.gridLayout_2.addWidget(self.label_subsetSpacing, 1, 0, 1, 1)

        self.spinBox_subSize = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_subSize.sizePolicy().hasHeightForWidth())
        self.spinBox_subSize.setSizePolicy(sizePolicy)
        self.spinBox_subSize.setMaximumSize(QtCore.QSize(60, 16777215))
        self.spinBox_subSize.setMinimum(1)
        self.spinBox_subSize.setObjectName("spinBox_subSize")
        self.gridLayout_2.addWidget(self.spinBox_subSize, 0, 1, 1, 1)

        self.label_px_subSize = QtWidgets.QLabel(self.centralwidget)
        self.label_px_subSize.setObjectName("label_px_subSize")
        self.gridLayout_2.addWidget(self.label_px_subSize, 0, 2, 1, 1)

        self.label_px_subSpacing = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_px_subSpacing.sizePolicy().hasHeightForWidth())
        self.label_px_subSpacing.setSizePolicy(sizePolicy)
        self.label_px_subSpacing.setObjectName("label_px_subSpacing")
        self.gridLayout_2.addWidget(self.label_px_subSpacing, 1, 2, 1, 1)

        self.label_subsetSize = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_subsetSize.sizePolicy().hasHeightForWidth())
        self.label_subsetSize.setSizePolicy(sizePolicy)
        self.label_subsetSize.setObjectName("label_subsetSize")
        self.gridLayout_2.addWidget(self.label_subsetSize, 0, 0, 1, 1)

        self.spinBox_subSpacing = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_subSpacing.sizePolicy().hasHeightForWidth())
        self.spinBox_subSpacing.setSizePolicy(sizePolicy)
        self.spinBox_subSpacing.setMaximumSize(QtCore.QSize(60, 16777215))
        self.spinBox_subSpacing.setMinimum(1)
        self.spinBox_subSpacing.setObjectName("spinBox_subSpacing")
        self.gridLayout_2.addWidget(self.spinBox_subSpacing, 1, 1, 1, 1)

        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 6, 1, 3)

        self.label_messages = QtWidgets.QLabel(self.centralwidget)
        self.label_messages.setIndent(0)
        self.label_messages.setObjectName("label_messages")
        self.gridLayout_3.addWidget(self.label_messages, 4, 6, 1, 1)

        self.plainTextEdit_messages = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_messages.setEnabled(True)
        self.plainTextEdit_messages.setReadOnly(False)
        self.plainTextEdit_messages.setMouseTracking(False)
        self.plainTextEdit_messages.setInputMethodHints(QtCore.Qt.ImhMultiLine)
        self.plainTextEdit_messages.setOverwriteMode(True)
        self.plainTextEdit_messages.setBackgroundVisible(False)
        self.plainTextEdit_messages.setMaximumSize(QtCore.QSize(16777215, 140))
        self.plainTextEdit_messages.setObjectName("plainTextEdit_messages")
        self.gridLayout_3.addWidget(self.plainTextEdit_messages, 5, 6, 1, 3)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(10, 0, -1, -1)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")

        self.label_arrowLength = QtWidgets.QLabel(self.centralwidget)
        self.label_arrowLength.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arrowLength.sizePolicy().hasHeightForWidth())
        self.label_arrowLength.setSizePolicy(sizePolicy)
        self.label_arrowLength.setObjectName("label_arrowLength")
        self.gridLayout.addWidget(self.label_arrowLength, 2, 1, 1, 2)

        self.lineEdit_arrowWidth = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_arrowWidth.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_arrowWidth.sizePolicy().hasHeightForWidth())
        self.lineEdit_arrowWidth.setSizePolicy(sizePolicy)
        self.lineEdit_arrowWidth.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_arrowWidth.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_arrowWidth.setObjectName("lineEdit_arrowWidth")
        self.gridLayout.addWidget(self.lineEdit_arrowWidth, 1, 3, 1, 3)

        self.label_arrowMin = QtWidgets.QLabel(self.centralwidget)
        self.label_arrowMin.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arrowMin.sizePolicy().hasHeightForWidth())
        self.label_arrowMin.setSizePolicy(sizePolicy)
        self.label_arrowMin.setMinimumSize(QtCore.QSize(50, 0))
        self.label_arrowMin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_arrowMin.setObjectName("label_arrowMin")
        self.gridLayout.addWidget(self.label_arrowMin, 6, 1, 1, 1)

        self.lineEdit_arrowMin = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_arrowMin.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_arrowMin.sizePolicy().hasHeightForWidth())
        self.lineEdit_arrowMin.setSizePolicy(sizePolicy)
        self.lineEdit_arrowMin.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_arrowMin.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_arrowMin.setObjectName("lineEdit_arrowMin")
        self.gridLayout.addWidget(self.lineEdit_arrowMin, 6, 2, 1, 1)

        self.checkBox_setLimits = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_setLimits.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_setLimits.sizePolicy().hasHeightForWidth())
        self.checkBox_setLimits.setSizePolicy(sizePolicy)
        self.checkBox_setLimits.setChecked(False)
        self.checkBox_setLimits.setObjectName("checkBox_setLimits")
        self.gridLayout.addWidget(self.checkBox_setLimits, 4, 1, 2, 5)

        self.lineEdit_arrowLength = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_arrowLength.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_arrowLength.sizePolicy().hasHeightForWidth())
        self.lineEdit_arrowLength.setSizePolicy(sizePolicy)
        self.lineEdit_arrowLength.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_arrowLength.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_arrowLength.setObjectName("lineEdit_arrowLength")
        self.gridLayout.addWidget(self.lineEdit_arrowLength, 2, 3, 1, 3)

        self.label_arrowWidth = QtWidgets.QLabel(self.centralwidget)
        self.label_arrowWidth.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arrowWidth.sizePolicy().hasHeightForWidth())
        self.label_arrowWidth.setSizePolicy(sizePolicy)
        self.label_arrowWidth.setObjectName("label_arrowWidth")
        self.gridLayout.addWidget(self.label_arrowWidth, 1, 1, 1, 2)

        self.lineEdit_arrowMax = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_arrowMax.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_arrowMax.sizePolicy().hasHeightForWidth())
        self.lineEdit_arrowMax.setSizePolicy(sizePolicy)
        self.lineEdit_arrowMax.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_arrowMax.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_arrowMax.setObjectName("lineEdit_arrowMax")
        self.gridLayout.addWidget(self.lineEdit_arrowMax, 6, 4, 1, 1)

        self.label_arrowMax = QtWidgets.QLabel(self.centralwidget)
        self.label_arrowMax.setVisible(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_arrowMax.sizePolicy().hasHeightForWidth())
        self.label_arrowMax.setSizePolicy(sizePolicy)
        self.label_arrowMax.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_arrowMax.setObjectName("label_arrowMax")
        self.gridLayout.addWidget(self.label_arrowMax, 6, 3, 1, 1)

        spacerItem8 = QtWidgets.QSpacerItem(20, 51, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem8, 0, 4, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(1, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem9, 2, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(1, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem10, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(1, 25, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem1, 6, 0, 1, 1)

        self.gridLayout_3.addLayout(self.gridLayout, 2, 8, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        app.aboutToQuit.connect(self.quit_program)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1496, 22))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.menuROI_and_Scaling = QtWidgets.QMenu(self.menubar)
        self.menuROI_and_Scaling.setObjectName("menuROI_and_Scaling")

        self.menuAnalysis = QtWidgets.QMenu(self.menubar)
        self.menuAnalysis.setToolTipsVisible(False)
        self.menuAnalysis.setObjectName("menuAnalysis")

        self.menuExport = QtWidgets.QMenu(self.menubar)
        self.menuExport.setObjectName("menuExport")

        self.menuBitmap_Image_jpg = QtWidgets.QMenu(self.menuExport)
        self.menuBitmap_Image_jpg.setObjectName("menuBitmap_Image_jpg")

        self.menuVector_Image_pdf = QtWidgets.QMenu(self.menuExport)
        self.menuVector_Image_pdf.setObjectName("menuVector_Image_pdf")

        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")

        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")

        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")

        self.actionSave_as = QtWidgets.QAction(MainWindow)
        self.actionSave_as.setObjectName("actionSave_as")

        self.actionImport_Images = QtWidgets.QAction(MainWindow)
        self.actionImport_Images.setObjectName("actionImport_Images")

        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")

        self.actionDraw_ROI = QtWidgets.QAction(MainWindow)
        self.actionDraw_ROI.setObjectName("actionDraw_ROI")

        self.actionScale_from_Image = QtWidgets.QAction(MainWindow)
        self.actionScale_from_Image.setObjectName("actionScale_from_Image")

        self.actionScale_from_Value = QtWidgets.QAction(MainWindow)
        self.actionScale_from_Value.setObjectName("actionScale_from_Value")

        self.actionSet_Parameters = QtWidgets.QAction(MainWindow)
        self.actionSet_Parameters.setObjectName("actionSet_Parameters")

        self.actionRUN_ANALYSIS = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.actionRUN_ANALYSIS.setFont(font)
        self.actionRUN_ANALYSIS.setObjectName("actionRUN_ANALYSIS")

        self.actionVideo = QtWidgets.QAction(MainWindow)
        self.actionVideo.setObjectName("actionVideo")

        self.actionSingle_Image_jpg = QtWidgets.QAction(MainWindow)
        self.actionSingle_Image_jpg.setObjectName("actionSingle_Image_jpg")

        self.actionAll_Images_jpg = QtWidgets.QAction(MainWindow)
        self.actionAll_Images_jpg.setObjectName("actionAll_Images_jpg")

        self.actionSingle_Image_pdf = QtWidgets.QAction(MainWindow)
        self.actionSingle_Image_pdf.setObjectName("actionSingle_Image_pdf")

        self.actionAll_Images_pdf = QtWidgets.QAction(MainWindow)
        self.actionAll_Images_pdf.setObjectName("actionAll_Images_pdf")

        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionImport_Images)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)

        self.menuROI_and_Scaling.addAction(self.actionDraw_ROI)
        self.menuROI_and_Scaling.addSeparator()
        self.menuROI_and_Scaling.addAction(self.actionScale_from_Image)
        self.menuROI_and_Scaling.addAction(self.actionScale_from_Value)

        self.menuAnalysis.addAction(self.actionSet_Parameters)
        self.menuAnalysis.addAction(self.actionRUN_ANALYSIS)

        self.menuBitmap_Image_jpg.addAction(self.actionSingle_Image_jpg)
        self.menuBitmap_Image_jpg.addAction(self.actionAll_Images_jpg)

        self.menuVector_Image_pdf.addAction(self.actionSingle_Image_pdf)
        self.menuVector_Image_pdf.addAction(self.actionAll_Images_pdf)

        self.menuExport.addAction(self.menuBitmap_Image_jpg.menuAction())
        self.menuExport.addAction(self.menuVector_Image_pdf.menuAction())
        self.menuExport.addAction(self.actionVideo)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuROI_and_Scaling.menuAction())
        self.menubar.addAction(self.menuAnalysis.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())

        self.start_new_project()
        self.retranslateUi(MainWindow)

        self.radioButton_quiver.toggled['bool'].connect(self.label_arrowWidth.setVisible)
        self.radioButton_quiver.toggled['bool'].connect(self.label_arrowLength.setVisible)
        self.radioButton_quiver.toggled['bool'].connect(self.checkBox_setLimits.setVisible)
        self.radioButton_quiver.toggled['bool'].connect(self.lineEdit_arrowLength.setVisible)
        self.radioButton_quiver.toggled['bool'].connect(self.lineEdit_arrowWidth.setVisible)
        self.checkBox_setLimits.toggled['bool'].connect(self.lineEdit_arrowMin.setVisible)
        self.checkBox_setLimits.toggled['bool'].connect(self.lineEdit_arrowMax.setVisible)
        self.checkBox_setLimits.toggled['bool'].connect(self.label_arrowMin.setVisible)
        self.checkBox_setLimits.toggled['bool'].connect(self.label_arrowMax.setVisible)
        self.radioButton_contour.toggled['bool'].connect(self.checkBox_setLimits.setVisible)
        self.radioButton_contour.toggled['bool'].connect(self.label_arrowMin.setVisible)
        self.radioButton_contour.toggled['bool'].connect(self.label_arrowMax.setVisible)
        self.radioButton_contour.toggled['bool'].connect(self.lineEdit_arrowMin.setVisible)
        self.radioButton_contour.toggled['bool'].connect(self.lineEdit_arrowMax.setVisible)
        self.radioButton_subsets.toggled['bool'].connect(self.label_arrowMin.setHidden)
        self.radioButton_subsets.toggled['bool'].connect(self.label_arrowMax.setHidden)
        self.radioButton_subsets.toggled['bool'].connect(self.lineEdit_arrowMin.setHidden)
        self.radioButton_subsets.toggled['bool'].connect(self.lineEdit_arrowMax.setHidden)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.setTabOrder(self.spinBox_subSize, self.spinBox_subSpacing)
        MainWindow.setTabOrder(self.spinBox_subSpacing, self.radioButton_showGrid)
        MainWindow.setTabOrder(self.radioButton_showGrid, self.radioButton_img)
        MainWindow.setTabOrder(self.radioButton_img, self.radioButton_mask)
        MainWindow.setTabOrder(self.radioButton_mask, self.radioButton_quiver)
        MainWindow.setTabOrder(self.radioButton_quiver, self.radioButton_contour)
        MainWindow.setTabOrder(self.radioButton_contour, self.radioButton_subsets)
        MainWindow.setTabOrder(self.radioButton_subsets, self.spinBox)
        MainWindow.setTabOrder(self.spinBox, self.lineEdit_arrowWidth)
        MainWindow.setTabOrder(self.lineEdit_arrowWidth, self.lineEdit_arrowLength)
        MainWindow.setTabOrder(self.lineEdit_arrowLength, self.checkBox_setLimits)
        MainWindow.setTabOrder(self.checkBox_setLimits, self.lineEdit_arrowMin)
        MainWindow.setTabOrder(self.lineEdit_arrowMin, self.lineEdit_arrowMax)
        MainWindow.setTabOrder(self.lineEdit_arrowMax, self.plainTextEdit_messages)

        # enable/disable widgets
        self.spinBox.setEnabled(False)
        self.menuROI_and_Scaling.setEnabled(False)
        self.menuAnalysis.setEnabled(False)
        self.menuExport.setEnabled(False)
        self.actionSave.setEnabled(False)
        self.actionSave_as.setEnabled(False)
        self.actionImport_Images.setEnabled(True)
        self.actionRUN_ANALYSIS.setEnabled(False)
        self.radioButton_img.setEnabled(False)
        self.radioButton_mask.setEnabled(False)
        self.radioButton_quiver.setEnabled(False)
        self.radioButton_contour.setEnabled(False)
        self.radioButton_subsets.setEnabled(False)
        self.radioButton_showGrid.setEnabled(False)
        self.label_subsetSize.setEnabled(False)
        self.spinBox_subSize.setEnabled(False)
        self.label_px_subSize.setEnabled(False)
        self.label_subsetSpacing.setEnabled(False)
        self.spinBox_subSpacing.setEnabled(False)
        self.label_px_subSpacing.setEnabled(False)
        self.progressBar.setVisible(False)

        # connect widgets to functions
        self.actionNew.triggered.connect(self.start_new_project)
        self.actionSet_Parameters.triggered.connect(self.open_dialog)
        self.actionOpen.triggered.connect(self.open_project)
        self.actionSave.triggered.connect(self.save_project)
        self.actionVideo.triggered.connect(self.export_video)
        self.actionSingle_Image_jpg.triggered.connect(self.export_jpg_single)
        self.actionAll_Images_jpg.triggered.connect(self.export_jpg_all)
        self.actionSingle_Image_pdf.triggered.connect(self.export_pdf_single)
        self.actionAll_Images_pdf.triggered.connect(self.export_pdf_all)
        self.actionQuit.triggered.connect(self.quit_program)
        self.actionSave_as.triggered.connect(self.open_saveas_dialog)
        self.actionImport_Images.triggered.connect(self.import_images)
        self.actionDraw_ROI.triggered.connect(self.choose_roi_image_and_draw_roi)
        self.actionScale_from_Image.triggered.connect(self.scale_from_image)
        self.actionScale_from_Value.triggered.connect(self.scale_from_value)
        self.actionRUN_ANALYSIS.triggered.connect(self.run_analysis)
        self.spinBox.valueChanged.connect(self.spin_box_value_changed)
        self.radioButton_mask.toggled.connect(self.update_gui)
        self.radioButton_showGrid.toggled.connect(self.update_gui)
        self.radioButton_quiver.toggled.connect(self.update_gui)
        self.radioButton_contour.toggled.connect(self.update_gui)
        self.radioButton_subsets.toggled.connect(self.update_gui)
        self.spinBox_subSpacing.setValue(self.variables['subset_spacing'])
        self.spinBox_subSize.setValue(self.variables['subset_size'])
        self.spinBox_subSpacing.valueChanged.connect(self.spin_box_sub_spacing_value_changed)
        self.spinBox_subSize.valueChanged.connect(self.spin_box_sub_size_value_changed)
        self.lineEdit_arrowWidth.textChanged.connect(self.arrow_width_value_changed)
        self.lineEdit_arrowLength.textChanged.connect(self.arrow_length_value_changed)
        self.lineEdit_arrowMin.textChanged.connect(self.arrow_min_value_changed)
        self.lineEdit_arrowMax.textChanged.connect(self.arrow_max_value_changed)
        self.checkBox_setLimits.toggled.connect(self.update_gui)
        # self.checkBox_showGrid.stateChanged.connect(self.update_grid)
        # self.lineEdit_subSize.textChanged.connect(self.update_grid)
        # self.lineEdit_subSpacing.textChanged.connect(self.update_grid)

    def on_window_resize(self, event):
        img_label_width = self.label_img.width()
        img_label_height = self.label_img.height()
        cur_img = cv2.imread(self.variables['image_names'][self.variables['current_image']])
        cur_img_width = cur_img.shape[1]
        cur_img_height = cur_img.shape[0]

    '''
        def rescale_image(self):
            my_image_height, my_image_width = self.gray_image.size
            factor = (self.gui_width/2)/my_image_width
            width = int(my_image_width*factor)
            height = int(my_image_height*factor)
            max_height = self.gui_height-self.space_for_text
            if height > max_height:
                factor = max_height/my_image_height
                height = int(my_image_height * factor)
                width = int(my_image_width*factor)
            return factor, (height, width)
        '''

    def switch_all_primary_widgets(self, switch_on):  # switches all the widgets visible after launching the GUI
        self.menuFile.setEnabled(switch_on)
        self.menuROI_and_Scaling.setEnabled(switch_on)
        self.menuAnalysis.setEnabled(switch_on)
        self.menuExport.setEnabled(switch_on)
        self.spinBox.setEnabled(switch_on)
        self.label_fileName.setEnabled(switch_on)
        self.radioButton_img.setEnabled(switch_on)
        self.radioButton_mask.setEnabled(switch_on)
        self.radioButton_showGrid.setEnabled(switch_on)
        self.label_subsetSize.setEnabled(switch_on)
        self.spinBox_subSize.setEnabled(switch_on)
        self.label_px_subSize.setEnabled(switch_on)
        self.label_subsetSpacing.setEnabled(switch_on)
        self.spinBox_subSpacing.setEnabled(switch_on)
        self.label_px_subSpacing.setEnabled(switch_on)
        self.label_messages.setEnabled(switch_on)
        self.plainTextEdit_messages.setEnabled(switch_on)
        # self.progressBar.setVisible(switch_on)

    def start_new_project(self):
        self.label_img.setPixmap(QPixmap('_aux/dummy_img.png'))
        # initial variables
        self.variables = {
            'program_just_started': True,
            'current_image': 0,
            'image_names': ['_aux/dummy_img.png'],
            'images_loaded': False,
            'roi_image': '',
            'project_name': 'not saved',
            'textfield_text': '',
            'project_saved': True,
            'project_already_saved': False,
            'save_roi_path': 'imgs/temp_roi',
            'save_results_path': '',
            'points_for_mask': [],
            'subset_spacing': 40,
            'subset_size': 30,
            'show_grid': False,
            'grid_xx': [],
            'grid_yy': [],
            'mask_reduced': [],
            'backward_analysis': False,
            'update_subset_positions': False,
            'frame_period': 1,
            'subpixel_accuracy': 20,
            'scale': 1,
            'image_size': [0, 0],  # [n_rows, n_cols]
            'quiver_arrow_width': 2,
            'quiver_arrow_length': 3,
            'plot_colorbar_min': 0,
            'plot_colorbar_max': 0.05,
            'calculation_finished': False,
            'result_image': ''
        }
        global subpixel_accuracy, frame_period, backward_analysis, update_subset_positions
        subpixel_accuracy = self.variables['subpixel_accuracy'] * 1
        frame_period = self.variables['frame_period'] * 1
        if self.variables['backward_analysis']:
            backward_analysis = True
        else:
            backward_analysis = False
        if self.variables['update_subset_positions']:
            update_subset_positions = True
        else:
            update_subset_positions = False
        self.plainTextEdit_messages.clear()
        self.switch_all_primary_widgets(False)
        self.menuFile.setEnabled(True)
        self.update_gui()

    def quit_program(self):
        if not self.variables['project_saved']:
            save_file = self.ask_to_save()
            if save_file:
                self.save_project()
        app.quit()

    def open_saveas_dialog(self):
        project_name, dialog_ok = QInputDialog.getText(None, 'Save as', 'Project name:')
        if dialog_ok:
            self.variables['project_name'] = '%s' % project_name
            if not os.path.exists('saved_projects/' + self.variables['project_name']):
                os.makedirs('saved_projects/' + self.variables['project_name'])
            if self.variables['images_loaded']:
                if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/source_images/'):
                    os.makedirs('saved_projects/' + self.variables['project_name'] + '/source_images/')
                for idx, im in enumerate(self.variables['image_names']):
                    _, filename = os.path.split(im)
                    shutil.copy(im, 'saved_projects/' + self.variables['project_name'] + '/source_images/' + filename)
                    self.variables['image_names'][idx] = 'saved_projects/' + self.variables['project_name'] + \
                                                         '/source_images/' + filename

            if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/roi'):
                os.makedirs('saved_projects/' + self.variables['project_name'] + '/roi')
            if len(self.variables['roi_image']) > 0:
                try:
                    if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/roi'):
                        os.makedirs('saved_projects/' + self.variables['project_name'] + '/roi')
                    shutil.copy(self.variables['save_roi_path'] + '/roi.jpg',
                                'saved_projects/' + self.variables['project_name'] + '/roi/roi.jpg')
                    shutil.copy(self.variables['save_roi_path'] + '/mask_bw.jpg',
                                'saved_projects/' + self.variables['project_name'] + '/roi/mask_bw.jpg')
                    shutil.copy(self.variables['save_roi_path'] + '/mask_col.jpg',
                                'saved_projects/' + self.variables['project_name'] + '/roi/mask_col.jpg')
                    shutil.copy(self.variables['save_roi_path'] + '/masked_roi.jpg',
                                'saved_projects/' + self.variables['project_name'] + '/roi/masked_roi.jpg')
                except shutil.SameFileError:
                    pass
            self.variables['save_roi_path'] = 'saved_projects/' + self.variables['project_name'] + '/roi'

            out_file = open('saved_projects/' + self.variables['project_name'] + '/variables.piv', 'w')
            out_file.write("%s" % self.variables)
            out_file.close()

            self.variables['project_saved'] = True
            self.variables['project_already_saved'] = True
            self.write_to_textfield('project saved')
            self.update_gui()

    def save_project(self):
        if not os.path.exists('saved_projects/' + self.variables['project_name']):
            os.makedirs('saved_projects/' + self.variables['project_name'])
        if self.variables['images_loaded']:
            if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/source_images/'):
                os.makedirs('saved_projects/' + self.variables['project_name'] + '/source_images/')
            for idx, im in enumerate(self.variables['image_names']):
                _, filename = os.path.split(im)
                try:
                    shutil.copy(im, 'saved_projects/' + self.variables['project_name'] + '/source_images/' + filename)
                except shutil.SameFileError:
                    pass

        out_file = open('saved_projects/' + self.variables['project_name'] + '/variables.piv', 'w')
        out_file.write("%s" % self.variables)
        out_file.close()

        self.variables['project_saved'] = True
        self.write_to_textfield('project saved')
        self.update_gui()

    def open_project(self):
        self.variables['program_just_started'] = True
        if not self.variables['project_saved']:
            save_file = self.ask_to_save()
            if save_file:
                self.save_project()
        folder_name = QFileDialog.getExistingDirectory(None, "Select project folder", os.getcwd() + "/saved_projects",
                                                       QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        try:
            in_file = open(folder_name + '/variables.piv', 'r')
            contents = in_file.read()
            self.variables = eval(contents)
            self.variables['project_saved'] = True
            self.variables['project_already_saved'] = True
            self.variables['current_image'] = 0
            self.plainTextEdit_messages.insertPlainText(self.variables['textfield_text'])
            self.plainTextEdit_messages.moveCursor(QTextCursor.End)
            self.write_to_textfield('project loaded')
            self.update_gui()
        except FileNotFoundError:
            pass
        self.variables['program_just_started'] = False
        self.variables['project_saved'] = True
        global subpixel_accuracy, frame_period, backward_analysis, update_subset_positions
        subpixel_accuracy = self.variables['subpixel_accuracy'] * 1
        frame_period = self.variables['frame_period'] * 1
        if self.variables['backward_analysis']:
            backward_analysis = True
        else:
            backward_analysis = False
        if self.variables['update_subset_positions']:
            update_subset_positions = True
        else:
            update_subset_positions = False

    def ask_to_save(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Do you want to save the current project?")
        msg.setWindowTitle("Save the project?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        retval = msg.exec_()
        return retval

    def import_images(self):
        self.variables['calculation_finished'] = False
        image_names = QFileDialog.getOpenFileNames(None, 'Import Images', os.getcwd() + '/imgs',
                                                   "Image files (*.jpg *.JPG *.jpeg *.JPEG *.gif "
                                                   "*.GIF *.png *.PNG)")[0]
        image_names.sort()
        if len(image_names) > 0:
            first_image = cv2.imread(image_names[0])
            first_image_h, first_image_w = len(first_image), len(first_image[0])
            imgs_same_size = True
            for im in image_names[1:]:
                cur_image = cv2.imread(im)
                cur_image_h, cur_image_w = len(cur_image), len(cur_image[0])
                if not (first_image_h == cur_image_h and first_image_w == cur_image_w):
                    imgs_same_size = False
                    self.write_to_textfield('!images must have the same dimensions!')
                    break
            if imgs_same_size:
                self.write_to_textfield('images imported successfully')
                self.variables['image_size'] = [first_image_h, first_image_w]
                self.variables['image_names'] = image_names[:]
                self.variables['images_loaded'] = True
        else:
            self.variables['images_loaded'] = False
            self.write_to_textfield('no images selected')
        self.variables['project_saved'] = False
        self.update_gui()

    def choose_roi_image_and_draw_roi(self):
        self.variables['calculation_finished'] = False
        self.variables['roi_image'] = QFileDialog.getOpenFileName(None, 'Import Images', os.getcwd() + '/imgs',
                                                                  "Image files (*.jpg *.JPG *.jpeg *.JPEG *.gif *.GIF "
                                                                  "*.png *.PNG)")[0]
        if self.variables['project_already_saved']:
            if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/roi'):
                os.makedirs('saved_projects/' + self.variables['project_name'] + '/roi')
            self.variables['save_roi_path'] = 'saved_projects/' + self.variables['project_name'] + '/roi'

        pts = []  # array for saving selected points

        def draw_roi(event, x, y, flags, param):
            img2 = img.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

            if event == cv2.EVENT_RBUTTONDOWN:
                pts.pop()

            if event == cv2.EVENT_LBUTTONDBLCLK:
                self.write_to_textfield('ROI saved, close using |Esc| key')
                mask = np.zeros(img.shape, np.uint8)
                points = np.array(pts, np.int32)
                points = points.reshape((-1, 1, 2))
                mask_points = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
                mask_bw = cv2.fillPoly(mask_points.copy(), [points], (255, 255, 255))  # white mask
                mask_red = cv2.fillPoly(mask_points.copy(), [points], (0, 0, 255))  # red mask
                show_image = cv2.addWeighted(src1=img, alpha=0.6, src2=mask_red, beta=0.4, gamma=0)  # mask overlay
                cv2.imwrite(self.variables['save_roi_path'] + '/roi.jpg', show_image)
                cv2.imwrite(self.variables['save_roi_path'] + '/mask_bw.jpg', mask_bw)
                cv2.imwrite(self.variables['save_roi_path'] + '/mask_col.jpg', mask)
                roi = cv2.bitwise_and(mask_bw, img)
                cv2.imwrite(self.variables['save_roi_path'] + '/masked_roi.jpg', roi)
                self.variables['points_for_mask'] = points.tolist()[:]
                return

            if len(pts) > 0:
                cv2.circle(img2, pts[-1], int(img.shape[0] / 200), (0, 0, 255), -1)

            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.circle(img2, pts[i], int(img.shape[0] / 160), (0, 0, 255), -1)
                    cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=1)

            cv2.imshow('ROI_image', img2)

        img = cv2.imread(self.variables['roi_image'])
        cv2.namedWindow('ROI_image', cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('ROI_image', draw_roi)  # calls the mouse callback function

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        cv2.destroyWindow('ROI_image')
        self.variables['project_saved'] = False
        self.update_gui()

    def scale_from_image(self):
        scale_image = QFileDialog.getOpenFileName(None, 'Import Images', os.getcwd() + '/imgs',
                                                  "Image files (*.jpg *.JPG *.jpeg *.JPEG *.gif *.GIF *.png *.PNG)")[0]
        self.variables['scale'] = get_scale(scale_image, self.variables['scale'])
        self.write_to_textfield('Scale set to %.4f px/unit_of_length' % self.variables['scale'])
        self.variables['project_saved'] = False
        self.update_gui()

    def scale_from_value(self):
        scale, ok_pressed = QInputDialog.getDouble(None, "Get scale for the measurements",
                                                   "Define the scale : [px per measurement unit (e.g., mm)]",
                                                   self.variables['scale'], 0, 1000, 5)
        if ok_pressed:
            self.variables['scale'] = scale * 1.0
        self.write_to_textfield('Scale set to %.4f px/unit_of_length' % self.variables['scale'])
        self.variables['project_saved'] = False
        self.update_gui()

    def spin_box_value_changed(self):
        self.variables['current_image'] = self.spinBox.value() - 1
        self.variables['project_saved'] = False
        self.update_gui()

    def spin_box_sub_size_value_changed(self):
        self.variables['calculation_finished'] = False
        self.variables['subset_size'] = self.spinBox_subSize.value()
        self.variables['project_saved'] = False
        self.update_gui()

    def spin_box_sub_spacing_value_changed(self):
        self.variables['calculation_finished'] = False
        self.variables['subset_spacing'] = self.spinBox_subSpacing.value()
        self.variables['project_saved'] = False
        self.update_gui()

    def arrow_width_value_changed(self):
        try:
            val = float(self.lineEdit_arrowWidth.text())
            self.variables['quiver_arrow_width'] = val * 1.0
        except:
            self.lineEdit_arrowWidth.setText(str(self.variables['quiver_arrow_width']))
        self.update_gui()

    def arrow_length_value_changed(self):
        try:
            val = float(self.lineEdit_arrowLength.text())
            self.variables['quiver_arrow_length'] = val * 1.0
        except:
            self.lineEdit_arrowLength.setText(str(self.variables['quiver_arrow_length']))
        self.update_gui()

    def arrow_min_value_changed(self):
        try:
            val = float(self.lineEdit_arrowMin.text())
            self.variables['plot_colorbar_min'] = val * 1.0
        except:
            self.lineEdit_arrowMin.setText(str(self.variables['plot_colorbar_min']))
        self.update_gui()

    def arrow_max_value_changed(self):
        try:
            val = float(self.lineEdit_arrowMax.text())
            self.variables['plot_colorbar_max'] = val * 1.0
        except:
            self.lineEdit_arrowMax.setText(str(self.variables['plot_colorbar_max']))
        self.update_gui()

    def open_dialog(self):
        self.variables['calculation_finished'] = False
        global subpixel_accuracy, frame_period, backward_analysis, update_subset_positions
        Dialog_setParameters = QtWidgets.QDialog()
        ui_setParameters = Ui_Dialog_setParameters()
        ui_setParameters.setupUi(Dialog_setParameters)
        if Dialog_setParameters.exec():
            dialog_outputs = (ui_setParameters.getInputs())
            frame_period, subpixel_accuracy, self.variables['backward_analysis'], \
            self.variables['update_subset_positions'] = dialog_outputs
            self.variables['frame_period'], self.variables['subpixel_accuracy'] = \
                float(frame_period), int(subpixel_accuracy)
            subpixel_accuracy = self.variables['subpixel_accuracy'] * 1
            frame_period = self.variables['frame_period'] * 1
            if self.variables['backward_analysis']:
                backward_analysis = True
            else:
                backward_analysis = False
            if self.variables['update_subset_positions']:
                update_subset_positions = True
            else:
                update_subset_positions = False

            if self.variables['backward_analysis']:
                params_to_string = 'backward analysis, '
            else:
                params_to_string = 'forward analysis, '
            if self.variables['update_subset_positions']:
                params_to_string += 'update of subset positions'
            else:
                params_to_string += 'subset positions fixed'
            self.write_to_textfield('analysis parameters set to: frame period %.2f, subpixel accuracy %d, %s' %
                                    (self.variables['frame_period'], self.variables['subpixel_accuracy'],
                                     params_to_string))
            self.variables['project_saved'] = False
            self.update_gui()

    def write_to_textfield(self, text_to_show):
        string_to_show = '\n' + datetime.now().strftime('%H:%M:%S - ')
        string_to_show += text_to_show
        self.variables['textfield_text'] += string_to_show
        self.plainTextEdit_messages.insertPlainText(string_to_show)
        self.plainTextEdit_messages.moveCursor(QTextCursor.End)

    def create_masked_current_image(self):
        pts = self.variables['points_for_mask'][:]
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        img_name = self.variables['image_names'][self.variables['current_image']]
        img = cv2.imread(img_name)
        mask = cv2.imread(self.variables['save_roi_path'] + '/mask_col.jpg')
        mask_points = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask_red = cv2.fillPoly(mask_points.copy(), [points], (0, 0, 255))  # red mask
        masked_image = cv2.addWeighted(src1=img, alpha=0.6, src2=mask_red, beta=0.4, gamma=0)  # mask overlay
        cv2.imwrite(self.variables['save_roi_path'] + '/cur_img_roi.jpg', masked_image)
        if len(self.variables['grid_xx']) == 0:
            self.create_grid_and_mask()

    def create_grid_and_mask(self):
        mask = rgb2gray(mpimg.imread(self.variables['save_roi_path'] + '/mask_bw.jpg'))
        thresh, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = mask.astype(bool)
        first_subset = np.ceil(self.variables['subset_spacing'] / 2)
        last_subset_x = self.variables['image_size'][1] - first_subset + 0.1
        last_subset_y = self.variables['image_size'][0]- first_subset + 0.1
        x_values = np.arange(first_subset, last_subset_x, self.variables['subset_spacing'])
        y_values = np.arange(first_subset, last_subset_y, self.variables['subset_spacing'])
        mask_reduced = np.zeros((len(y_values), len(x_values)), dtype=bool)
        for ix, x_value in enumerate(x_values):
            for iy, y_value in enumerate(y_values):
                if mask[int(y_value)][int(x_value)]:
                    mask_reduced[iy][ix] = True
        xx, yy = np.meshgrid(x_values, y_values)
        self.variables['grid_xx'] = xx.tolist()[:]
        self.variables['grid_yy'] = yy.tolist()[:]
        self.variables['mask_reduced'] = mask_reduced.tolist()[:]

    def plot_grid(self):
        mask = rgb2gray(mpimg.imread(self.variables['save_roi_path'] + '/mask_bw.jpg'))
        thresh, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = mask.astype(bool)
        temp_image = rgb2gray(plt.imread(self.variables['image_names'][self.variables['current_image']]))
        image_masked = 255. * (temp_image / 255.) ** 0.15
        image_masked[mask] = temp_image[mask]
        xx = np.array(self.variables['grid_xx'])
        yy = np.array(self.variables['grid_yy'])
        mask_reduced = np.array(self.variables['mask_reduced'])
        xx_1d = xx.flatten()
        yy_1d = yy.flatten()
        mask_1d = mask_reduced.flatten()
        fig = plt.figure()
        plt.imshow(image_masked, cmap='gray')
        plt.axis('off')
        hs = 0.5 * self.variables['subset_size']
        for i in range(len(xx_1d)):
            if mask_1d[i]:
                plt.plot(xx_1d[i], yy_1d[i], '+', color='lawngreen', markersize=1, linewidth=0.7)
                x_coords = (xx_1d[i] - hs, xx_1d[i] + hs, xx_1d[i] + hs, xx_1d[i] - hs, xx_1d[i] - hs)
                y_coords = (yy_1d[i] - hs, yy_1d[i] - hs, yy_1d[i] + hs, yy_1d[i] + hs, yy_1d[i] - hs)
                plt.plot(x_coords, y_coords, '-', color='lawngreen', linewidth=0.7)
        plt.draw()
        plt.savefig(self.variables['save_roi_path'] + '/cur_grid.jpg', bbox_inches='tight', pad_inches = 0, dpi=400)
        plt.clf()
        plt.close(fig)

    def run_analysis(self):
        if not self.variables['project_already_saved']:
            self.write_to_textfield('project must be saved before running the analysis')
            self.open_saveas_dialog()
            self.write_to_textfield('analysis started')
        if not os.path.exists('saved_projects/' + self.variables['project_name'] + '/results'):
            os.makedirs('saved_projects/' + self.variables['project_name'] + '/results')
        self.variables['save_results_path'] = 'saved_projects/' + self.variables['project_name'] + '/results'

        self.switch_all_primary_widgets(False)
        start_time = datetime.now().replace(microsecond=0)
        self.write_to_textfield('analysis started')
        self.progressBar.setVisible(True)
        self.progressBar.setRange(0, len(self.variables['image_names']))

        # create grid
        self.create_grid_and_mask()

        # calculate displacements
        paths = self.variables['image_names'][:]
        if self.variables['backward_analysis']:
            paths.sort(reverse=True)
        times = []
        counter = 0
        ref_image = plt.imread(paths[0])
        grid_xx_backup = self.variables['grid_xx'][:]
        grid_yy_backup = self.variables['grid_yy'][:]
        remainder_x = np.zeros_like(self.variables['mask_reduced'])
        remainder_y = np.zeros_like(self.variables['mask_reduced'])
        for f in paths:
            counter += 1
            times.append(counter * self.variables['frame_period'])
            cur_image = plt.imread(f)
            cur_xx, cur_yy, cur_dx, cur_dy = [], [], [], []
            for iy in range(len(self.variables['grid_xx'])):
                for ix in range(len(self.variables['grid_xx'][0])):
                    if self.variables['mask_reduced'][iy][ix]:
                        node = [self.variables['grid_xx'][iy][ix], self.variables['grid_yy'][iy][ix]]
                        cur_xx.append(self.variables['grid_xx'][iy][ix])
                        cur_yy.append(self.variables['grid_yy'][iy][ix])
                        min_x, max_x, min_y, max_y = get_boundaries(node, self.variables['subset_size'],
                                                                    self.variables['image_size'][1],
                                                                    self.variables['image_size'][0])
                        subset_ref = ref_image[min_y:max_y, min_x:max_x, :]
                        subset_cur = cur_image[min_y:max_y, min_x:max_x, :]
                        shift, _, _ = phase_cross_correlation(subset_ref, subset_cur,
                                                              upsample_factor=self.variables['subpixel_accuracy'])
                        x_displacement = -1 * shift[1]
                        y_displacement = -1 * shift[0]
                        cur_dx.append(x_displacement)
                        cur_dy.append(y_displacement)

                        if self.variables['update_subset_positions']:
                            self.variables['grid_xx'][iy][ix] += int(x_displacement)
                            self.variables['grid_yy'][iy][ix] += int(y_displacement)
                            cur_remainder_x = x_displacement - int(x_displacement)
                            cur_remainder_y = y_displacement - int(y_displacement)
                            remainder_x[iy][ix] += cur_remainder_x
                            remainder_y[iy][ix] += cur_remainder_y
                            if abs(remainder_x[iy][ix]) > 1.0:
                                self.variables['grid_xx'][iy][ix] += int(remainder_x[iy][ix])
                                remainder_x[iy][ix] -= int(remainder_x[iy][ix])
                            if abs(remainder_y[iy][ix]) > 1.0:
                                self.variables['grid_yy'][iy][ix] += int(remainder_y[iy][ix])
                                remainder_y[iy][ix] -= int(remainder_y[iy][ix])
            ref_image = cur_image * 1.0

            np.savetxt(self.variables['save_results_path'] + '/x-coords_%04d.txt' % (counter - 1), cur_xx, fmt='%.9f')
            np.savetxt(self.variables['save_results_path'] + '/y-coords_%04d.txt' % (counter - 1), cur_yy, fmt='%.9f')
            np.savetxt(self.variables['save_results_path'] + '/dx_%04d.txt' % (counter - 1), cur_dx, fmt='%.9f')
            np.savetxt(self.variables['save_results_path'] + '/dy_%04d.txt' % (counter - 1), cur_dy, fmt='%.9f')
            self.progressBar.setValue(counter)

        self.variables['grid_xx'] = grid_xx_backup[:]
        self.variables['grid_yy'] = grid_yy_backup[:]
        np.savetxt(self.variables['save_results_path'] + '/times.txt', times, fmt='%.9f')
        np.savetxt(self.variables['save_results_path'] + '/imageNames.txt', self.variables['image_names'], fmt='%s')
        np.savetxt(self.variables['save_results_path'] + '/scale.txt', [self.variables['scale']], fmt='%.12f')
        np.savetxt(self.variables['save_results_path'] + '/framePeriod.txt', [self.variables['frame_period']], fmt='%.12f')
        np.savetxt(self.variables['save_results_path'] + '/subsetSize.txt', [self.variables['subset_size']], fmt='%d')
        np.savetxt(self.variables['save_results_path'] + '/subsetSpacing.txt', [self.variables['subset_spacing']], fmt='%d')
        if self.variables['backward_analysis']:
            np.savetxt(self.variables['save_results_path'] + '/backwardCalculation.txt', [1], fmt='%d')
        else:
            np.savetxt(self.variables['save_results_path'] + '/backwardCalculation.txt', [0], fmt='%d')

        self.variables['calculation_finished'] = True
        self.variables['project_saved'] = False
        finish_time = datetime.now().replace(microsecond=0)
        self.progressBar.setVisible(False)
        self.switch_all_primary_widgets(True)
        duration = finish_time - start_time
        hours, minutes, seconds = duration.days * 24 + duration.seconds // 3600, (duration.seconds % 3600) // 60, \
                                  (duration.seconds % 3600) % 60
        self.write_to_textfield('analysis finished after %02d:%02d:%02d' % (hours, minutes, seconds))
        self.update_gui()

    def export_video(self):
        self.export_results(save_single=False, save_raster=True, save_video=True)

    def export_jpg_single(self):
        self.export_results(save_single=True, save_raster=True, save_video=False)

    def export_jpg_all(self):
        self.export_results(save_single=False, save_raster=True, save_video=False)

    def export_pdf_single(self):
        self.export_results(save_single=True, save_raster=False, save_video=False)

    def export_pdf_all(self):
        self.export_results(save_single=False, save_raster=False, save_video=False)

    def export_results(self, save_single, save_raster, save_video):
        self.write_to_textfield('export started')
        if not os.path.exists('export/' + self.variables['project_name']):
            os.makedirs('export/' + self.variables['project_name'])
        if save_raster:
            if not os.path.exists('export/' + self.variables['project_name'] + '/jpg'):
                os.makedirs('export/' + self.variables['project_name'] + '/jpg')
            save_path = 'export/' + self.variables['project_name'] + '/jpg'
        else:
            if not os.path.exists('export/' + self.variables['project_name'] + '/pdf'):
                os.makedirs('export/' + self.variables['project_name'] + '/pdf')
            save_path = 'export/' + self.variables['project_name'] + '/pdf'
        if save_video:
            if not os.path.exists('export/' + self.variables['project_name'] + '/avi'):
                os.makedirs('export/' + self.variables['project_name'] + '/avi')
            save_video_path = 'export/' + self.variables['project_name'] + '/avi'

        if self.radioButton_subsets.isChecked():
            save_file = 'subsets'
        elif self.radioButton_contour.isChecked():
            save_file = 'velocity_contour'
        else:
            save_file = 'velocity_quiver'

        self.switch_all_primary_widgets(False)
        start_time = datetime.now().replace(microsecond=0)
        self.progressBar.setVisible(True)
        self.progressBar.setRange(0, len(self.variables['image_names']))

        # export images
        if save_single:
            self.create_plot(save_jpg=save_raster)
            if save_raster:
                full_path = save_path + '/' + save_file + '_%04d.jpg' % (self.variables['current_image'] + 1)
                shutil.copy2(self.variables['save_results_path'] + '/currentResultImage.jpg', full_path)
            else:
                full_path = save_path + '/' + save_file + '_%04d.pdf' % (self.variables['current_image'] + 1)
                shutil.copy2(self.variables['save_results_path'] + '/currentResultImage.pdf', full_path)
        else:
            full_paths = []
            for i in range(len(self.variables['image_names'])):
                self.progressBar.setValue(i)
                self.variables['current_image'] = i * 1
                self.create_plot(save_jpg=save_raster)
                if save_raster:
                    full_path = save_path + '/' + save_file + '_%04d.jpg' % (self.variables['current_image'] + 1)
                    shutil.copy2(self.variables['save_results_path'] + '/currentResultImage.jpg', full_path)
                else:
                    full_path = save_path + '/' + save_file + '_%04d.pdf' % (self.variables['current_image'] + 1)
                    shutil.copy2(self.variables['save_results_path'] + '/currentResultImage.pdf', full_path)
                full_paths.append(full_path)
        if save_video:
            full_video_path = save_video_path + '/' + save_file + '.avi'
            create_video(full_paths, full_video_path, video_duration=len(self.variables['image_names'])/2)
            # vytvorit funkce pro jednotlive moznosti v menu, kde se bude volat funkce export_results

        finish_time = datetime.now().replace(microsecond=0)
        self.progressBar.setVisible(False)
        self.switch_all_primary_widgets(True)
        duration = finish_time - start_time
        hours, minutes, seconds = duration.days * 24 + duration.seconds // 3600, (duration.seconds % 3600) // 60, \
                                  (duration.seconds % 3600) % 60
        self.write_to_textfield('export finished after %02d:%02d:%02d' % (hours, minutes, seconds))
        self.create_plot(save_jpg=True)
        self.variables['project_saved'] = False
        self.update_gui()

    def create_plot(self, save_jpg):
        quiver_arrow_width = self.variables['quiver_arrow_width'] / 1000
        quiver_arrow_length = self.variables['quiver_arrow_length'] * 1000
        if self.checkBox_setLimits.isChecked():
            quiver_fix_limits = True
        else:
            quiver_fix_limits = False
        quiver_fixed_limits = [self.variables['plot_colorbar_min'], self.variables['plot_colorbar_max']]
        data_loaded = False
        try:
            scale_factor = np.loadtxt(self.variables['save_results_path'] + '/scale.txt', dtype=float)
            cur_frame_period = np.loadtxt(self.variables['save_results_path'] + '/framePeriod.txt', dtype=float)
            subset_size = np.loadtxt(self.variables['save_results_path'] + '/subsetSize.txt', dtype=int)
            cur_xx = np.loadtxt(self.variables['save_results_path'] + '/x-coords_%04d.txt' %
                                self.variables['current_image'], dtype=float)
            cur_yy = np.loadtxt(self.variables['save_results_path'] + '/y-coords_%04d.txt' %
                                self.variables['current_image'], dtype=float)
            cur_dx = np.loadtxt(self.variables['save_results_path'] + '/dx_%04d.txt' %
                                self.variables['current_image'], dtype=float)
            cur_dy = np.loadtxt(self.variables['save_results_path'] + '/dy_%04d.txt' %
                                self.variables['current_image'], dtype=float)
            data_loaded = True
        except OSError:
            self.write_to_textfield('the results could not be loaded')
        if data_loaded:
            if self.variables['backward_analysis'] == 1:
                cur_dx *= -1
                cur_dy *= -1
            cur_dx /= scale_factor
            cur_dy /= scale_factor
            cur_vx = cur_dx / cur_frame_period
            cur_vy = cur_dy / cur_frame_period
            cur_vtot = np.sqrt(cur_vx ** 2 + cur_vy ** 2)
            if self.radioButton_subsets.isChecked():
                plot_deformed_grid(self.variables['image_names'][self.variables['current_image']],
                                   self.variables['save_results_path'] + '/currentResultImage',
                                   cur_xx, cur_yy, subset_size, save_jpg)
            elif self.radioButton_contour.isChecked():
                plot_contours_irregular_grid(self.variables['image_names'][self.variables['current_image']],
                                             self.variables['save_results_path'] + '/currentResultImage',
                                             cur_xx, cur_yy, cur_vtot, quiver_fix_limits, quiver_fixed_limits, save_jpg)
            else:
                plot_quiver(self.variables['image_names'][self.variables['current_image']],
                            self.variables['save_results_path'] + '/currentResultImage',
                            cur_xx, cur_yy, cur_vx, cur_vy, cur_vtot, quiver_arrow_width, quiver_arrow_length,
                            quiver_fix_limits, quiver_fixed_limits, save_jpg)

    def update_gui(self):
        if self.variables['calculation_finished']:
            self.radioButton_quiver.setEnabled(True)
            self.radioButton_contour.setEnabled(True)
            self.radioButton_subsets.setEnabled(True)
            self.menuExport.setEnabled(True)

        if self.variables['images_loaded']:
            if self.radioButton_img.isChecked():
                self.label_img.setPixmap(QPixmap(self.variables['image_names'][self.variables['current_image']]))
            elif self.radioButton_mask.isChecked():
                self.create_masked_current_image()
                self.label_img.setPixmap(QPixmap(self.variables['save_roi_path'] + '/cur_img_roi.jpg'))
            elif self.radioButton_showGrid.isChecked():
                self.create_grid_and_mask()
                self.plot_grid()
                self.label_img.setPixmap(QPixmap(self.variables['save_roi_path'] + '/cur_grid.jpg'))
            else:
                self.create_plot(save_jpg=True)
                self.label_img.setPixmap(QPixmap(self.variables['save_results_path'] + '/currentResultImage.jpg'))

            self.label_fileName.setText("Filename: %s" % self.variables['image_names'][self.variables['current_image']])
            self.spinBox.setEnabled(True)
            self.spinBox.setMaximum(len(self.variables['image_names']))
            self.spinBox.setProperty("value", self.variables['current_image'] + 1)
            self.spinBox_subSize.setProperty("value", self.variables['subset_size'])
            self.spinBox_subSpacing.setProperty("value", self.variables['subset_spacing'])
            self.spinBox.setProperty("value", self.variables['current_image'] + 1)
            if self.variables['project_saved']:
                self.main_window_handle.setWindowTitle("PyPIV [%s]" % self.variables['project_name'])
            else:
                self.main_window_handle.setWindowTitle("PyPIV [%s*]" % self.variables['project_name'])
            if self.variables['project_already_saved']:
                if self.variables['project_saved']:
                    self.actionSave.setEnabled(False)
                else:
                    self.actionSave.setEnabled(True)
            if self.variables['images_loaded']:
                self.actionSave_as.setEnabled(True)
                self.menuROI_and_Scaling.setEnabled(True)
                self.menuAnalysis.setEnabled(True)
            if len(self.variables['roi_image']) > 0:
                self.actionRUN_ANALYSIS.setEnabled(True)
                self.radioButton_img.setEnabled(True)
                self.radioButton_mask.setEnabled(True)
                self.radioButton_showGrid.setEnabled(True)
                self.label_subsetSize.setEnabled(True)
                self.spinBox_subSize.setEnabled(True)
                self.label_px_subSize.setEnabled(True)
                self.label_subsetSpacing.setEnabled(True)
                self.spinBox_subSpacing.setEnabled(True)
                self.label_px_subSpacing.setEnabled(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.radioButton_img.setStatusTip(_translate("MainWindow", "Show only the image"))
        self.radioButton_img.setText(_translate("MainWindow", "Image"))
        self.radioButton_mask.setStatusTip(_translate("MainWindow", "Show the image with a mask representing ROI"))
        self.radioButton_mask.setText(_translate("MainWindow", "Mask"))
        self.radioButton_showGrid.setStatusTip(_translate("MainWindow", "Show a grid within the image"))
        self.radioButton_showGrid.setText(_translate("MainWindow", "Grid"))
        self.radioButton_quiver.setStatusTip(_translate("MainWindow", "Show the image with a quiver plot"))
        self.radioButton_quiver.setText(_translate("MainWindow", "Quiver plot"))
        self.radioButton_contour.setStatusTip(_translate("MainWindow", "Show the image with a contour plot"))
        self.radioButton_contour.setText(_translate("MainWindow", "Contour plot"))
        self.radioButton_subsets.setStatusTip(_translate("MainWindow", "Show the image with subsets"))
        self.radioButton_subsets.setText(_translate("MainWindow", "Subsets"))
        self.spinBox.setStatusTip(_translate("MainWindow", "Number of the current image"))
        self.progressBar.setStatusTip(_translate("MainWindow", "Progress of the running analysis"))
        self.label_fileName.setStatusTip(_translate("MainWindow", "Name of the current image"))
        self.label_fileName.setText(_translate("MainWindow", ""))
        self.label_subsetSpacing.setStatusTip(_translate("MainWindow", "Set the subset spacing for the analysis"))
        self.label_subsetSpacing.setText(_translate("MainWindow", "Subset spacing:"))
        self.radioButton_showGrid.setStatusTip(_translate("MainWindow", "Show a grid within the image"))
        self.radioButton_showGrid.setText(_translate("MainWindow", "Grid"))
        self.spinBox_subSize.setStatusTip(_translate("MainWindow", "Set the subset size for the analysis"))
        self.label_px_subSize.setStatusTip(_translate("MainWindow", "Set the subset size for the analysis"))
        self.label_px_subSize.setText(_translate("MainWindow", "px"))
        self.label_px_subSpacing.setStatusTip(_translate("MainWindow", "Set the subset spacing for the analysis"))
        self.label_px_subSpacing.setText(_translate("MainWindow", "px"))
        self.label_subsetSize.setStatusTip(_translate("MainWindow", "Set the subset size for the analysis"))
        self.label_subsetSize.setText(_translate("MainWindow", "Subset size:"))
        self.spinBox_subSpacing.setStatusTip(_translate("MainWindow", "Set the subset spacing for the analysis"))
        self.label_messages.setStatusTip(_translate("MainWindow", "Message box"))
        self.label_messages.setText(_translate("MainWindow", "Messages:"))
        self.plainTextEdit_messages.setStatusTip(_translate("MainWindow", "Message box"))
        self.plainTextEdit_messages.setPlainText(_translate("MainWindow", "              >>> WELCOME TO PyPIV <<<"))
        self.plainTextEdit_messages.moveCursor(QTextCursor.End)
        self.label_arrowLength.setStatusTip(_translate("MainWindow", "Set the length of the arrow in the quiver plot"))
        self.label_arrowLength.setText(_translate("MainWindow", "Arrow length:"))
        self.lineEdit_arrowWidth.setStatusTip(_translate("MainWindow", "Set the width of the arrow in the quiver plot"))
        # self.lineEdit_arrowWidth.setText(_translate("MainWindow", "2"))
        self.label_arrowMin.setStatusTip(_translate("MainWindow", "Required minimum value for plotting colors"))
        self.label_arrowMin.setText(_translate("MainWindow", "min"))
        self.lineEdit_arrowMin.setStatusTip(_translate("MainWindow", "Required minimum value for plotting colors"))
        # self.lineEdit_arrowMin.setText(_translate("MainWindow", "1"))
        self.checkBox_setLimits.setStatusTip(_translate("MainWindow", "Set the limits manually to prevent skewing due to extreme values"))
        self.checkBox_setLimits.setText(_translate("MainWindow", "Set limits"))
        self.checkBox_setLimits.setChecked(False)
        self.lineEdit_arrowLength.setStatusTip(_translate("MainWindow", "Set the length of the arrow in the quiver plot"))
        # self.lineEdit_arrowLength.setText(_translate("MainWindow", "3"))
        self.label_arrowWidth.setStatusTip(_translate("MainWindow", "Set the width of the arrow in the quiver plot"))
        self.label_arrowWidth.setText(_translate("MainWindow", "Arrow width:"))
        self.lineEdit_arrowMax.setStatusTip(_translate("MainWindow", "Required maximum value for plotting colors"))
        # self.lineEdit_arrowMax.setText(_translate("MainWindow", "5"))
        self.lineEdit_arrowWidth.setText(str(self.variables['quiver_arrow_width']))
        self.lineEdit_arrowLength.setText(str(self.variables['quiver_arrow_length']))
        self.lineEdit_arrowMin.setText(str(self.variables['plot_colorbar_min']))
        self.lineEdit_arrowMax.setText(str(self.variables['plot_colorbar_max']))
        self.label_arrowMax.setStatusTip(_translate("MainWindow", "Required maximum value for plotting colors"))
        self.label_arrowMax.setText(_translate("MainWindow", "max"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuROI_and_Scaling.setTitle(_translate("MainWindow", "ROI and Scaling"))
        self.menuAnalysis.setTitle(_translate("MainWindow", "Analysis"))
        self.menuExport.setTitle(_translate("MainWindow", "Export"))
        self.menuBitmap_Image_jpg.setStatusTip(_translate("MainWindow", "Export bitmap image with chosen plot type"))
        self.menuBitmap_Image_jpg.setTitle(_translate("MainWindow", "Bitmap Image (*.jpg)"))
        self.menuVector_Image_pdf.setStatusTip(_translate("MainWindow", "Export vector image with chosen plot type"))
        self.menuVector_Image_pdf.setTitle(_translate("MainWindow", "Vector Image (*.pdf)"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionNew.setStatusTip(_translate("MainWindow", "Create and name new project"))
        self.actionNew.setWhatsThis(_translate("MainWindow", "Create and name new project"))
        self.actionNew.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setStatusTip(_translate("MainWindow", "Open a saved project from a folder"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save the current project"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSave_as.setText(_translate("MainWindow", "Save As"))
        self.actionSave_as.setStatusTip(_translate("MainWindow", "Save the project as a different one"))
        self.actionSave_as.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionImport_Images.setText(_translate("MainWindow", "Import Images"))
        self.actionImport_Images.setStatusTip(_translate("MainWindow", "Import images for analysis"))
        self.actionImport_Images.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setStatusTip(_translate("MainWindow", "Quit the program"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionDraw_ROI.setText(_translate("MainWindow", "Draw ROI"))
        self.actionDraw_ROI.setStatusTip(_translate("MainWindow", "Define the region of interest by clicking within the image"))
        self.actionScale_from_Image.setText(_translate("MainWindow", "Scale from Image"))
        self.actionScale_from_Image.setStatusTip(_translate("MainWindow", "Relate pixels to milimeters by clicking within the image"))
        self.actionScale_from_Value.setText(_translate("MainWindow", "Scale from Value"))
        self.actionScale_from_Value.setStatusTip(_translate("MainWindow", "Relate pixels to milimeters by setting a value"))
        self.actionSet_Parameters.setText(_translate("MainWindow", "Set Parameters"))
        self.actionSet_Parameters.setStatusTip(_translate("MainWindow", "Set analysis type and its parameters"))
        self.actionSet_Parameters.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionRUN_ANALYSIS.setText(_translate("MainWindow", "RUN ANALYSIS"))
        self.actionRUN_ANALYSIS.setStatusTip(_translate("MainWindow", "Run the analysis"))
        self.actionRUN_ANALYSIS.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.actionVideo.setText(_translate("MainWindow", "Video"))
        self.actionVideo.setStatusTip(_translate("MainWindow", "Export all the images as a video"))
        self.actionSingle_Image_jpg.setText(_translate("MainWindow", "Single Step"))
        self.actionSingle_Image_jpg.setStatusTip(_translate("MainWindow", "Export the current image as a bitmap"))
        self.actionAll_Images_jpg.setText(_translate("MainWindow", "All Steps"))
        self.actionAll_Images_jpg.setStatusTip(_translate("MainWindow", "Export all the images as a bitmap"))
        self.actionSingle_Image_pdf.setText(_translate("MainWindow", "Single Step"))
        self.actionSingle_Image_pdf.setStatusTip(_translate("MainWindow", "Export the current image as a vector image"))
        self.actionAll_Images_pdf.setText(_translate("MainWindow", "All Steps"))
        self.actionAll_Images_pdf.setStatusTip(_translate("MainWindow", "Export all the images as vector images"))


class Ui_Dialog_setParameters(object):
    def setupUi(self, Dialog_setParameters):
        Dialog_setParameters.setObjectName("Dialog_setParameters")
        Dialog_setParameters.resize(240, 180)
        Dialog_setParameters.setStatusTip("")
        Dialog_setParameters.setSizeGripEnabled(False)

        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog_setParameters)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.label_framePeriod = QtWidgets.QLabel(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_framePeriod.sizePolicy().hasHeightForWidth())
        self.label_framePeriod.setSizePolicy(sizePolicy)
        self.label_framePeriod.setObjectName("label_framePeriod")
        self.gridLayout.addWidget(self.label_framePeriod, 0, 0, 1, 1)

        self.label_unit1px = QtWidgets.QLabel(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unit1px.sizePolicy().hasHeightForWidth())
        self.label_unit1px.setSizePolicy(sizePolicy)
        self.label_unit1px.setObjectName("label_unit1px")
        self.gridLayout.addWidget(self.label_unit1px, 1, 2, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)

        self.label_subpixelAccuracy = QtWidgets.QLabel(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_subpixelAccuracy.sizePolicy().hasHeightForWidth())
        self.label_subpixelAccuracy.setSizePolicy(sizePolicy)
        self.label_subpixelAccuracy.setObjectName("label_subpixelAccuracy")
        self.gridLayout.addWidget(self.label_subpixelAccuracy, 1, 0, 1, 1)

        self.checkBox_backwardAnalysis = QtWidgets.QCheckBox(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_backwardAnalysis.sizePolicy().hasHeightForWidth())
        self.checkBox_backwardAnalysis.setSizePolicy(sizePolicy)
        self.checkBox_backwardAnalysis.setStatusTip("")
        self.checkBox_backwardAnalysis.setChecked(True)
        self.checkBox_backwardAnalysis.setObjectName("checkBox_backwardAnalysis")
        self.gridLayout.addWidget(self.checkBox_backwardAnalysis, 4, 0, 1, 2)

        self.lineEdit_subpixelAccuracy = QtWidgets.QLineEdit(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_subpixelAccuracy.sizePolicy().hasHeightForWidth())
        self.lineEdit_subpixelAccuracy.setSizePolicy(sizePolicy)
        self.lineEdit_subpixelAccuracy.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_subpixelAccuracy.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_subpixelAccuracy.setObjectName("lineEdit_subpixelAccuracy")
        self.gridLayout.addWidget(self.lineEdit_subpixelAccuracy, 1, 1, 1, 1)

        self.checkBox_updatePosition = QtWidgets.QCheckBox(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_updatePosition.sizePolicy().hasHeightForWidth())
        self.checkBox_updatePosition.setSizePolicy(sizePolicy)
        self.checkBox_updatePosition.setStatusTip("")
        self.checkBox_updatePosition.setChecked(True)
        self.checkBox_updatePosition.setObjectName("checkBox_updatePosition")
        self.gridLayout.addWidget(self.checkBox_updatePosition, 5, 0, 1, 2)

        self.lineEdit_framePeriod = QtWidgets.QLineEdit(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_framePeriod.sizePolicy().hasHeightForWidth())
        self.lineEdit_framePeriod.setSizePolicy(sizePolicy)
        self.lineEdit_framePeriod.setMaximumSize(QtCore.QSize(35, 16777215))
        self.lineEdit_framePeriod.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_framePeriod.setObjectName("lineEdit_framePeriod")
        self.gridLayout.addWidget(self.lineEdit_framePeriod, 0, 1, 1, 1)

        self.label_unitS = QtWidgets.QLabel(Dialog_setParameters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unitS.sizePolicy().hasHeightForWidth())
        self.label_unitS.setSizePolicy(sizePolicy)
        self.label_unitS.setObjectName("label_unitS")
        self.gridLayout.addWidget(self.label_unitS, 0, 2, 1, 1)

        self.line = QtWidgets.QFrame(Dialog_setParameters)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 3, 0, 1, 3)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog_setParameters)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply|QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(Dialog_setParameters)

        self.buttonBox.clicked.connect(Dialog_setParameters.accept)
        self.buttonBox.rejected.connect(Dialog_setParameters.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog_setParameters)

        Dialog_setParameters.setTabOrder(self.lineEdit_framePeriod, self.lineEdit_subpixelAccuracy)
        Dialog_setParameters.setTabOrder(self.lineEdit_subpixelAccuracy, self.checkBox_backwardAnalysis)
        Dialog_setParameters.setTabOrder(self.checkBox_backwardAnalysis, self.checkBox_updatePosition)


    def retranslateUi(self, Dialog_setParameters):
        _translate = QtCore.QCoreApplication.translate
        Dialog_setParameters.setWindowTitle(_translate("Dialog_setParameters", "Analysis Parameters"))
        self.label_framePeriod.setText(_translate("Dialog_setParameters", "Frame period:"))
        self.label_unit1px.setText(_translate("Dialog_setParameters", "1/px"))
        self.label_subpixelAccuracy.setText(_translate("Dialog_setParameters", "Subpixel accuracy:"))
        self.checkBox_backwardAnalysis.setText(_translate("Dialog_setParameters", "Backward analysis"))
        self.checkBox_backwardAnalysis.setChecked(backward_analysis)
        self.lineEdit_subpixelAccuracy.setText(_translate("Dialog_setParameters", str(subpixel_accuracy)))
        self.checkBox_updatePosition.setText(_translate("Dialog_setParameters", "Update subset position"))
        self.checkBox_updatePosition.setChecked(update_subset_positions)
        self.lineEdit_framePeriod.setText(_translate("Dialog_setParameters", str(frame_period)))
        self.label_unitS.setText(_translate("Dialog_setParameters", ""))

    # new function
    def getInputs(self):
        return (self.lineEdit_framePeriod.text(), self.lineEdit_subpixelAccuracy.text(),
                self.checkBox_backwardAnalysis.isChecked(), self.checkBox_updatePosition.isChecked())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
