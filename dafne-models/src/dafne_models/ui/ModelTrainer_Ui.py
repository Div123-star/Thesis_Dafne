# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ModelTrainer.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ModelTrainerUI(object):
    def setupUi(self, ModelTrainerUI):
        ModelTrainerUI.setObjectName("ModelTrainerUI")
        ModelTrainerUI.resize(824, 824)
        self.verticalLayout = QtWidgets.QVBoxLayout(ModelTrainerUI)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(ModelTrainerUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.location_Text = QtWidgets.QLineEdit(ModelTrainerUI)
        self.location_Text.setEnabled(False)
        self.location_Text.setReadOnly(True)
        self.location_Text.setObjectName("location_Text")
        self.horizontalLayout.addWidget(self.location_Text)
        self.choose_Button = QtWidgets.QPushButton(ModelTrainerUI)
        self.choose_Button.setObjectName("choose_Button")
        self.horizontalLayout.addWidget(self.choose_Button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.transfer_learning_checkBox = QtWidgets.QGroupBox(ModelTrainerUI)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.transfer_learning_checkBox.setFont(font)
        self.transfer_learning_checkBox.setCheckable(True)
        self.transfer_learning_checkBox.setChecked(False)
        self.transfer_learning_checkBox.setObjectName("transfer_learning_checkBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.transfer_learning_checkBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.transfer_learning_checkBox)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.BaseModel_Text = QtWidgets.QLineEdit(self.transfer_learning_checkBox)
        self.BaseModel_Text.setEnabled(False)
        self.BaseModel_Text.setReadOnly(True)
        self.BaseModel_Text.setObjectName("BaseModel_Text")
        self.horizontalLayout_6.addWidget(self.BaseModel_Text)
        self.pushButton = QtWidgets.QPushButton(self.transfer_learning_checkBox)
        self.pushButton.setEnabled(False)
        self.pushButton.setAutoDefault(True)
        self.pushButton.setDefault(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_6.addWidget(self.pushButton)
        self.fine_tune_label = QtWidgets.QLabel(self.transfer_learning_checkBox)
        self.fine_tune_label.setObjectName("fine_tune_label")
        self.horizontalLayout_6.addWidget(self.fine_tune_label)
        self.fine_tune_at_spinBox = QtWidgets.QSpinBox(self.transfer_learning_checkBox)
        self.fine_tune_at_spinBox.setMaximum(300)
        self.fine_tune_at_spinBox.setObjectName("fine_tune_at_spinBox")
        self.horizontalLayout_6.addWidget(self.fine_tune_at_spinBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.verticalLayout.addWidget(self.transfer_learning_checkBox)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(ModelTrainerUI)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.model_location_Text = QtWidgets.QLineEdit(ModelTrainerUI)
        self.model_location_Text.setEnabled(False)
        self.model_location_Text.setReadOnly(True)
        self.model_location_Text.setObjectName("model_location_Text")
        self.horizontalLayout_2.addWidget(self.model_location_Text)
        self.save_choose_Button = QtWidgets.QPushButton(ModelTrainerUI)
        self.save_choose_Button.setEnabled(True)
        self.save_choose_Button.setDefault(True)
        self.save_choose_Button.setObjectName("save_choose_Button")
        self.horizontalLayout_2.addWidget(self.save_choose_Button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.force_preprocess_check = QtWidgets.QCheckBox(ModelTrainerUI)
        self.force_preprocess_check.setEnabled(False)
        self.force_preprocess_check.setCheckable(True)
        self.force_preprocess_check.setChecked(True)
        self.force_preprocess_check.setObjectName("force_preprocess_check")
        self.horizontalLayout_3.addWidget(self.force_preprocess_check)
        self.advanced_button = QtWidgets.QPushButton(ModelTrainerUI)
        self.advanced_button.setEnabled(True)
        self.advanced_button.setObjectName("advanced_button")
        self.horizontalLayout_3.addWidget(self.advanced_button)
        self.advanced_widget = QtWidgets.QWidget(ModelTrainerUI)
        self.advanced_widget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.advanced_widget.sizePolicy().hasHeightForWidth())
        self.advanced_widget.setSizePolicy(sizePolicy)
        self.advanced_widget.setMinimumSize(QtCore.QSize(0, 0))
        self.advanced_widget.setObjectName("advanced_widget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.advanced_widget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.advanced_widget)
        self.label_3.setEnabled(True)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.levels_spin = QtWidgets.QSpinBox(self.advanced_widget)
        self.levels_spin.setEnabled(True)
        self.levels_spin.setMinimum(1)
        self.levels_spin.setProperty("value", 5)
        self.levels_spin.setObjectName("levels_spin")
        self.horizontalLayout_5.addWidget(self.levels_spin)
        self.label_4 = QtWidgets.QLabel(self.advanced_widget)
        self.label_4.setEnabled(True)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.convlayers_spin = QtWidgets.QSpinBox(self.advanced_widget)
        self.convlayers_spin.setEnabled(True)
        self.convlayers_spin.setProperty("value", 2)
        self.convlayers_spin.setObjectName("convlayers_spin")
        self.horizontalLayout_5.addWidget(self.convlayers_spin)
        self.label_5 = QtWidgets.QLabel(self.advanced_widget)
        self.label_5.setEnabled(True)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.kernsize_spin = QtWidgets.QSpinBox(self.advanced_widget)
        self.kernsize_spin.setEnabled(True)
        self.kernsize_spin.setMinimum(1)
        self.kernsize_spin.setProperty("value", 2)
        self.kernsize_spin.setObjectName("kernsize_spin")
        self.horizontalLayout_5.addWidget(self.kernsize_spin)
        self.horizontalLayout_3.addWidget(self.advanced_widget)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.progressBar = QtWidgets.QProgressBar(ModelTrainerUI)
        self.progressBar.setEnabled(False)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout.addWidget(self.progressBar)
        self.progress_label = QtWidgets.QLabel(ModelTrainerUI)
        self.progress_label.setEnabled(True)
        self.progress_label.setObjectName("progress_label")
        self.verticalLayout.addWidget(self.progress_label)
        self.fit_output_box = QtWidgets.QGroupBox(ModelTrainerUI)
        self.fit_output_box.setEnabled(True)
        self.fit_output_box.setMinimumSize(QtCore.QSize(800, 400))
        self.fit_output_box.setObjectName("fit_output_box")
        self.verticalLayout.addWidget(self.fit_output_box)
        spacerItem = QtWidgets.QSpacerItem(20, 45, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.preprocess_Button = QtWidgets.QPushButton(ModelTrainerUI)
        self.preprocess_Button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preprocess_Button.sizePolicy().hasHeightForWidth())
        self.preprocess_Button.setSizePolicy(sizePolicy)
        self.preprocess_Button.setObjectName("preprocess_Button")
        self.horizontalLayout_4.addWidget(self.preprocess_Button)
        self.fit_Button = QtWidgets.QPushButton(ModelTrainerUI)
        self.fit_Button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fit_Button.sizePolicy().hasHeightForWidth())
        self.fit_Button.setSizePolicy(sizePolicy)
        self.fit_Button.setObjectName("fit_Button")
        self.horizontalLayout_4.addWidget(self.fit_Button)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.retranslateUi(ModelTrainerUI)
        QtCore.QMetaObject.connectSlotsByName(ModelTrainerUI)

    def retranslateUi(self, ModelTrainerUI):
        _translate = QtCore.QCoreApplication.translate
        ModelTrainerUI.setWindowTitle(_translate("ModelTrainerUI", "Form"))
        self.label.setText(_translate("ModelTrainerUI", "Data location"))
        self.choose_Button.setText(_translate("ModelTrainerUI", "Choose..."))
        self.transfer_learning_checkBox.setTitle(_translate("ModelTrainerUI", "Transfer Learning"))
        self.label_6.setText(_translate("ModelTrainerUI", "Base model:"))
        self.pushButton.setText(_translate("ModelTrainerUI", "Choose..."))
        self.fine_tune_label.setText(_translate("ModelTrainerUI", "fine_tune_at"))
        self.label_2.setText(_translate("ModelTrainerUI", "Output model:"))
        self.save_choose_Button.setText(_translate("ModelTrainerUI", "Choose..."))
        self.force_preprocess_check.setText(_translate("ModelTrainerUI", "Force preprocess"))
        self.advanced_button.setText(_translate("ModelTrainerUI", "Advanced settings"))
        self.label_3.setText(_translate("ModelTrainerUI", "Levels: "))
        self.label_4.setText(_translate("ModelTrainerUI", " Conv. layers: "))
        self.label_5.setText(_translate("ModelTrainerUI", " Kernel size: "))
        self.progress_label.setText(_translate("ModelTrainerUI", "Text"))
        self.fit_output_box.setTitle(_translate("ModelTrainerUI", "Fitting output"))
        self.preprocess_Button.setText(_translate("ModelTrainerUI", "Preprocess only"))
        self.fit_Button.setText(_translate("ModelTrainerUI", "Preprocess + Fit model"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ModelTrainerUI = QtWidgets.QWidget()
    ui = Ui_ModelTrainerUI()
    ui.setupUi(ModelTrainerUI)
    ModelTrainerUI.show()
    sys.exit(app.exec_())
