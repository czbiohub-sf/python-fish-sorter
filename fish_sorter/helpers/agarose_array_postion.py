import json
import sys
import os
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QMessageBox

class AgaroseArray(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Form layout for input fields
        form_layout = QFormLayout()

        self.rows_input = QLineEdit()
        self.columns_input = QLineEdit()
        self.row_spacing_input = QLineEdit()
        self.column_spacing_input = QLineEdit()
        self.length_input = QLineEdit()
        self.width_input = QLineEdit()
        self.shape_input = QComboBox()
        self.shape_input.addItems(["circular", "rectangular"])

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

    def define_wells(self):
        try:
            rows = int(self.rows_input.text())
            columns = int(self.columns_input.text())
            row_spacing = float(self.row_spacing_input.text())
            column_spacing = float(self.column_spacing_input.text())
            length = float(self.length_input.text())
            width = float(self.width_input.text())
            shape = self.shape_input.currentText()

            if any(value <= 0 for value in [rows, columns, row_spacing, column_spacing, length, width]):
                raise ValueError

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

            date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            array_file = f"{num_wells}{shape}_array_{date_stamp}.json"
            array_dir = Path().absolute().parent / "configs/arrays"
            array_path = os.path.join(array_dir, array_file)

            with open(array_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            QMessageBox.information(self, 'Success', f'Values are valid and have been saved to {array_path}')
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Please enter valid numbers greater than 0 for all fields.')
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
                well_names.append(f"{row_label}{c + 1}")
        return well_names

    def generate_well_coordinates(self, rows, columns, row_spacing, column_spacing, length, width):
        x_spacing = column_spacing + length
        y_spacing = row_spacing + width
        well_coordinates = [[(c * x_spacing, r * y_spacing) for c in range(columns)] for r in range(rows)]
        return well_coordinates

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AgaroseArray()
    ex.show()
    sys.exit(app.exec_())