# Import necessary Kivy and KivyMD modules
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.relativelayout import MDRelativeLayout
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.image import Image
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.label import MDLabel
from kivy.lang import Builder

# Other necessary Python modules
import threading
import json
import os


# Load the KV file
Builder.load_file("./assets/design.kv")


# Path to the configuration file
JSON_CONFIG_FILE_PATH = "./config.json"


# Define classes for different layouts
class DropFileLayout(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._color_yellow_ = [1, 0.85, 0.25, 1]


class LoadingLayout(MDRelativeLayout):
    pass


class ModelInfoLayout(MDBoxLayout):
    pass


# Main application class
class PneumoniaDetectionApp(MDApp):

    def build(self):
        # Load the configuration file and apply the window theme "Dark"/"Light"
        self.load_config()
        self.title = "XRray Pneumonia Detection"
        self.theme_cls = ThemeManager()
        self.theme_cls.theme_style = self.window_theme

        # Show the loading layout while the background loading takes place
        self.loading_layout = LoadingLayout()
        loading_thread = threading.Thread(target=self.backgroundLoading)
        loading_thread.start()

        # Initialize the main layout
        self.main_layout = DropFileLayout()
        self.myIds = self.main_layout.ids

        # Bind the event that will allow us to drop files (images) to the window
        Window.bind(on_drop_file=self.on_file_selected)

        # Set the output layout to the loading layout initially
        self.output_layout = self.loading_layout

        return self.output_layout

    # Load the configuration from the JSON file
    def load_config(self):
        def get_global_path(local_path):
            # Get the absolute path of the current working directory
            current_dir = os.getcwd()

            # Join the current directory with the local path to get the global path
            global_path = os.path.join(current_dir, local_path)

            return global_path

        try:
            with open(JSON_CONFIG_FILE_PATH) as config_file:
                self.config_file = json.load(config_file)

                # Extract necessary values from the config file

                self.model_filename = self.config_file['model']['model_file_path']
                self.full_model_path = get_global_path(self.model_filename)
                self.labels = self.config_file['model']['labels']

                self.window_width = self.config_file["window"]["window_width"]
                self.window_height = self.config_file["window"]["window_height"]
                self.window_theme = self.config_file["window"]["window_theme"]

                self.model_performance_train_loss = str(
                    self.config_file["train"]["loss"])
                self.model_performance_train_acc = str(
                    self.config_file["train"]["accuracy"])
                self.model_performance_test_loss = str(
                    self.config_file["test"]["loss"])
                self.model_performance_test_acc = str(
                    self.config_file["test"]["accuracy"])
                self.model_performance_val_loss = str(
                    self.config_file["val"]["loss"])
                self.model_performance_val_acc = str(
                    self.config_file["val"]["accuracy"])
                self.model_performance_trained_epochs = str(
                    self.config_file["other"]["trained_epochs"])
                self.model_performance_trained_params = str(
                    self.config_file["other"]["trained_params"])

                # Set the window size
                Window.size = (self.window_width, self.window_height)

        except Exception as err:
            # Show a MDSnackbar and print the error if there's any issue with the configuration file
            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()
            print(f"There is an error: {err}")

            # Set default window size if there's an error
            Window.size = (800, 600)

    # Method to change the current layout of the app
    def changeCurrentWindow(self, new_layout):
        self.output_layout.clear_widgets()
        self.output_layout.add_widget(new_layout)

    # Background loading at the beginning
    def backgroundLoading(self):

        # Import required modules (cv2, numpy, tensorflow) in the background
        # while displaying a loading screen on the Kivy main window

        # Importing modules inside the function may take some time,
        # so we do it in the background while showing a loading screen.
        import cv2
        import numpy as np
        import tensorflow as tf
        import datetime
        import matplotlib.pyplot as plt

        # Make the imported modules accessible across the whole class
        # by storing them as class attributes (acting like global modules)

        # Since we cannot make them global within the function's local scope,
        # we create class attributes to store these modules for access across the whole class.
        self.modules_cv2 = cv2
        self.modules_tf = tf
        self.modules_np = np
        self.modules_datetime = datetime
        self.modules_plt = plt

        self.modules_plt.axis("off")

        # Load the model using TensorFlow
        self.model = self.modules_tf.keras.models.load_model(
            self.full_model_path, compile=False)

        # Schedule the opening of the main window after loading the model
        # (Note: We are using Kivy's Clock to schedule the function)
        Clock.schedule_once(
            lambda dt: self.changeCurrentWindow(self.main_layout))

    # Method called when a file is selected (dropped) onto the app
    def on_file_selected(self, instance, selection, a, b):
        try:
            self.absolute_image_path = selection.decode("utf-8")

            self.myIds["_selected_image_"].source = self.absolute_image_path
            print("Selected file:", self.absolute_image_path)
        except Exception as err:

            print(
                ":ERROR| PROBABLY, cannot load an image. Ensure image has a correct format.")
            print(f"There is an error: {err}")

            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()

        self.analyzeImage()

    # Method to analyze the selected image
    def analyzeImage(self, instance=None):

        try:
            self.current_image_array = self.modules_cv2.imread(
                self.absolute_image_path, self.modules_cv2.IMREAD_GRAYSCALE)

            self.current_image_array = self.modules_cv2.resize(
                self.current_image_array, (400, 300))

            self.current_expanded_image = self.modules_tf.expand_dims(
                self.current_image_array, axis=-1)
            self.current_expanded_image = self.modules_tf.expand_dims(
                self.current_expanded_image, axis=0)

            probas = self.model.predict(self.current_expanded_image)[0]
            predicted_class_idx = self.modules_np.argmax(probas)
            self.predicted_class = self.labels[predicted_class_idx]

            def probaFeedback(proba):
                val = round(proba, 2)
                if val == 0:
                    return "0 (NO)"
                return "1 (YES)"

            print("_"*50)
            print(f"Probailities: \t{probas}")
            print(f"Predicted class index: {predicted_class_idx}")
            print(f"Predicted class: \t{self.predicted_class}")
            print("_"*50)

            self.predicted_normal_proba = round(probas[0], 2)
            self.predicted_pneumonia_proba = round(probas[1], 2)

            self.myIds["_label_normal_percentage_"].text = probaFeedback(
                self.predicted_normal_proba)
            self.myIds["_label_pneumonia_percentage_"].text = probaFeedback(
                self.predicted_pneumonia_proba)
            self.myIds["_label_class_"].text = str(self.predicted_class)

            MDSnackbar(
                MDLabel(text=f"SUCCESS| Image loaded FROM {self.absolute_image_path}")).open()

        except Exception as err:
            print(f"There is an error: {err}")
            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()

    # Method to show model information layout (button event)
    def btnModelInfo(self, instance=None):

        self.model_info_layout = ModelInfoLayout()

        try:

            self.model_info_layout.ids["_label_model_name_"].text = self.model_filename
            self.model_info_layout.ids["_label_full_model_path_"].text = self.full_model_path
            self.model_info_layout.ids["_label_model_train_loss_"].text = self.model_performance_train_loss
            self.model_info_layout.ids["_label_model_train_acc_"].text = self.model_performance_train_acc
            self.model_info_layout.ids["_label_model_test_loss_"].text = self.model_performance_test_loss
            self.model_info_layout.ids["_label_model_test_acc_"].text = self.model_performance_test_acc
            self.model_info_layout.ids["_label_model_val_loss_"].text = self.model_performance_val_loss
            self.model_info_layout.ids["_label_model_val_acc_"].text = self.model_performance_val_acc
            self.model_info_layout.ids["_label_model_trained_epochs_"].text = self.model_performance_trained_epochs
            self.model_info_layout.ids["_label_model_trained_params_"].text = self.model_performance_trained_params

        except Exception as err:
            print(f"There is an error: {err}")
            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()

        self.changeCurrentWindow(self.model_info_layout)

    # Method to return to the main window from the model information layout (button event)
    def btnReturnMainWindow(self, instance=None):
        self.changeCurrentWindow(self.main_layout)

    # Method to analyze the selected image again (button event)
    def btnAnalyzeAgain(self, instance=None):
        self.analyzeImage()
        print("Analyzed Again")

    # Method to save the analyzed image with exported results
    def saveImage(self):

        self.modules_cv2.imwrite(self.exportFilename, self.exportedPNG)

        def show_file_manager(instance=None):
            absolute_home_path = os.path.expanduser("~")
            self.file_manager.show(absolute_home_path)

        def exit_manager(*args):
            self.file_manager.close()
            os.remove(self.exportFilename)

        def select_path(path):
            if path:
                save_path = os.path.join(path, self.exportFilename)
                self.image_widget.export_to_png(save_path)
                path_snackbar = MDSnackbar(
                    MDLabel(text=f"SUCCESS| Image saved AS {save_path}")).open()
            exit_manager()

        try:

            self.image_path = self.exportFilename

            self.file_manager = MDFileManager(
                exit_manager=exit_manager,
                select_path=select_path,
            )

            self.image_widget = Image(source=self.image_path, size=(
                self.window_height, self.window_width), fit_mode="contain", keep_ratio=True)

            show_file_manager()

        except Exception as err:
            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()
            print(f"There is an error: {err}")

    # Method to export the results of image analysis (button event)
    def btnExportResults(self, instance=None):
        self.current_date = self.modules_datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

        try:

            self.notes = ["_"*15,
                          f"RESULT: {self.predicted_class}",
                          "_"*15,
                          f"DATE: {self.current_date}",
                          "_"*15,
                          "MODEL PERFORMANCE: (loss, accuracy)",
                          f"      Train:  {self.model_performance_train_loss}, {self.model_performance_train_acc}",
                          f"      Test:   {self.model_performance_test_loss}, {self.model_performance_test_acc}",
                          f"      Val:    {self.model_performance_val_loss}, {self.model_performance_val_acc}"
                          ]

            self.exportedPNG = self.exportPNG(
                self.current_image_array, self.notes)
            self.exportFilename = f"X-ray_Pneumonia_Detection_{self.current_date}.png"

            self.saveImage()

        except Exception as err:
            print(f"No Image Loaded OR error is {err}")
            MDSnackbar(MDLabel(text=f"ERROR| {err}")).open()

    # Method to draw a text data on the image

    def exportPNG(self, img_array, notes):
        text_canvas = self.modules_np.zeros(
            shape=img_array.shape, dtype=img_array.dtype)

        def add_multiline_notes_to_image(img, notes):
            # Define the font and text properties
            font = self.modules_cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            font_color = (243, 213, 1)  # Green color (BGR format)

            # Calculate the starting position for multiline text
            x, y = 10, 20
            line_height = 30  # Adjust the line height as needed

            # Add multiline text annotations (notes) to the image
            for line in notes:
                self.modules_cv2.putText(
                    img, line, (x, y), font, font_scale, font_color, font_thickness)
                y += line_height

            return img

        text_canvas = add_multiline_notes_to_image(text_canvas, notes)

        ready_image = self.modules_np.vstack((img_array, text_canvas))

        return ready_image


# Entry point of the application
if __name__ == "__main__":
    # Run the Kivy App
    kivy_app = PneumoniaDetectionApp()
    kivy_app.run()

    try:
        os.remove(kivy_app.exportFilename)
    except:
        pass
