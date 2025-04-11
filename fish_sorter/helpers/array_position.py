import json
import sys
import os
from datetime import datetime
from pathlib import Path
from qtpy.QtWidgets import (
    QApplication,  
    QComboBox,
    QDoubleSpinBox,
    QFormLayout, 
    QLabel, 
    QLineEdit, 
    QMessageBox,
    QPushButton, 
    QSpinBox,
    QVBoxLayout,
    QWidget
)

class GenerateArray(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Form layout for input fields
        form_layout = QFormLayout()

        self.rows_input = QSpinBox()
        self.columns_input = QSpinBox()
        self.row_spacing_input = QDoubleSpinBox()
        self.set_spinbox(self.row_spacing_input)
        self.column_spacing_input = QDoubleSpinBox()
        self.set_spinbox(self.column_spacing_input)
        self.length_input = QDoubleSpinBox()
        self.set_spinbox(self.length_input)
        self.width_input = QDoubleSpinBox()
        self.set_spinbox(self.width_input)
        self.shape_input = QComboBox()
        self.shape_input.addItems(["circular_array", "rectangular_array", "well_plate"])

        form_layout.addRow(QLabel('Rows:'), self.rows_input)
        form_layout.addRow(QLabel('Columns:'), self.columns_input)
        form_layout.addRow(QLabel('Row Spacing [um]:'), self.row_spacing_input)
        form_layout.addRow(QLabel('Column Spacing [um]:'), self.column_spacing_input)
        form_layout.addRow(QLabel('Slot Length [um]:'), self.length_input)
        form_layout.addRow(QLabel('Slot Width [um]:'), self.width_input)
        form_layout.addRow(QLabel('Well Shape:'), self.shape_input)

        layout.addLayout(form_layout)

        # Submit button
        submit_button = QPushButton('Create agarose array position file')
        submit_button.clicked.connect(self.define_wells)
        layout.addWidget(submit_button)

        self.setLayout(layout)
        self.setWindowTitle('Define Agarose Array')

    def set_spinbox(self, spinbox):
        """
        Used to set the SpinBox parameters

        :param spinbox: Qt spinbox to update
        :type spinbox: QDoubleSpinBox
        """

        spinbox.setRange(0, 150000.00)
        spinbox.setSingleStep(0.05)
        spinbox.setDecimals(2)
        spinbox.setSuffix(" um")
    
    def define_wells(self):
        try:
            rows = self.rows_input.value()
            columns = self.columns_input.value()
            row_spacing = self.row_spacing_input.value()
            column_spacing = self.column_spacing_input.value()
            length = self.length_input.value()
            width = self.width_input.value()
            shape = self.shape_input.currentText()

            array_design = {
                'rows': rows,
                'columns': columns,
                'row_spacing': row_spacing,
                'column_spacing': column_spacing,
                'slot_length': length,
                'slot_width': width,
                'well_shape': shape
            }

            num_wells = rows * columns
            well_names = self.generate_well_names(rows, columns)
            well_coordinates = self.generate_well_coordinates(rows, columns, row_spacing, column_spacing, length, width)

            well_def = {
                'total_wells': num_wells,
                'well_names': well_names,
                'well_coordinates': well_coordinates
            }

            data = {
                'array_design': array_design,
                'wells': well_def
            }

            date_stamp = datetime.now().strftime("%Y%m%d")
            array_file = f"{num_wells}{shape}{date_stamp}.json"
            array_dir = Path().absolute().parent / "python-fish-sorter/fish_sorter/configs/arrays"
            array_path = os.path.join(array_dir, array_file)

            with open(array_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            QMessageBox.information(self, 'Success', f'Values are valid and have been saved to {array_path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'An error occurred: {str(e)}')

    def generate_well_names(self, rows, columns):
        def get_column_name(n):
            name = ''
            while n > 0:
                n, remainder = divmod(n - 1, 26)
                name = chr(65 + remainder) + name
            return name

        well_names = []
        for r in range(rows):
            row_label = get_column_name(r + 1)
            for c in range(columns):
                col_label = f"{c + 1:02}"
                well_names.append(f"{row_label}{col_label}")
        return well_names

    def generate_well_coordinates(self, rows, columns, row_spacing, column_spacing, length, width):
        x_spacing = column_spacing + length
        y_spacing = row_spacing + width
        well_coordinates = [[(c * x_spacing, r * y_spacing) for c in range(columns)] for r in range(rows)]
        return well_coordinates

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GenerateArray()
    ex.show()
    sys.exit(app.exec_())