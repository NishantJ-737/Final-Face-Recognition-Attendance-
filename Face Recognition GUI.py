import json
from kivy.config import Config
Config.set('graphics', 'resizable', '0')  # Disable window resizing

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView  # Import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
import face_recognition
from datetime import datetime, timedelta
import os
import pandas as pd
import shutil
import threading
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
import tkinter as tk
from tkinter import filedialog
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Rectangle, Line

class FaceRecognitionApp(App):
    TIME_SLOTS_FILE = 'time_slots.json'
    RECOGNITION_THRESHOLD = 3  # Number of consecutive recognitions required to mark attendance

    def load_time_slots(self):
        try:
            with open(self.TIME_SLOTS_FILE, 'r') as file:
                time_slots = json.load(file)
                self.entry_start_time = time_slots.get('entry_start_time', '09:00 AM')
                self.entry_stop_time = time_slots.get('entry_stop_time', '09:15 AM')
                self.exit_start_time = time_slots.get('exit_start_time', '05:00 PM')
                self.exit_stop_time = time_slots.get('exit_stop_time', '05:30 PM')
        except FileNotFoundError:
            self.entry_start_time = '09:00 AM'
            self.entry_stop_time = '09:15 AM'
            self.exit_start_time = '05:00 PM'
            self.exit_stop_time = '05:30 PM'
        print(f"Time slots loaded: Entry {self.entry_start_time} to {self.entry_stop_time}, Exit {self.exit_start_time} to {self.exit_stop_time}")

    def save_time_slots(self):
        time_slots = {
            'entry_start_time': self.entry_start_time,
            'entry_stop_time': self.entry_stop_time,
            'exit_start_time': self.exit_start_time,
            'exit_stop_time': self.exit_stop_time
        }
        with open(self.TIME_SLOTS_FILE, 'w') as file:
            json.dump(time_slots, file)
        print("Time slots saved.")

    def build(self):
        self.load_time_slots()

        self.entry_start_input = None
        self.entry_stop_input = None
        self.exit_start_input = None
        self.exit_stop_input = None
        self.path = 'images'
        self.images = []
        self.classNames = []
        self.myList = os.listdir(self.path)
        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            if curImg is not None:
                curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
                self.images.append(curImg)
                self.classNames.append(os.path.splitext(cl)[0])
                print(f"Loaded and processed image: {cl}")
            else:
                print(f"Warning: Unable to read image '{cl}'.")

        self.encodeListKnown = self.findEncodings(self.images)
        print(f"Found {len(self.encodeListKnown)} encodings.")
        
        self.cap = None
        self.is_camera_running = False
        self.frame = None

        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.title_label = Label(text='Face Recognition System', font_size=24, bold=True, size_hint=(1, 0.1))
        self.layout.add_widget(self.title_label)

        self.result_label = Label(text='Recognition Result: Unknown', font_size=20, halign='center', color=(0, 0.7, 0.7, 1), size_hint=(1, 0.1))
        self.layout.add_widget(self.result_label)

        self.image_widget = Image(size_hint=(1, 0.6))
        self.layout.add_widget(self.image_widget)

        self.control_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.1))
        self.start_button = Button(text='Start Camera', on_press=self.toggle_camera, font_size=20, size_hint=(0.2, 1))
        self.control_layout.add_widget(self.start_button)

        self.view_summary_button = Button(text='View Attendance Summary', on_press=self.view_summary, font_size=20, size_hint=(0.4, 1))
        self.download_button = Button(text='Download Attendance', on_press=self.download_attendance, font_size=20, size_hint=(0.3, 1))
        self.settings_button = Button(text='Settings', on_press=self.open_settings, font_size=20, size_hint=(0.2, 1))
        
        self.control_layout.add_widget(self.view_summary_button)
        self.control_layout.add_widget(self.download_button)
        self.control_layout.add_widget(self.settings_button)

        self.layout.add_widget(self.control_layout)

        self.history_label = Label(text='Recognition History:', font_size=20, halign='left', color=(0, 0.7, 0.7, 1), size_hint=(1, 0.1))
        self.layout.add_widget(self.history_label)

        self.history_text = Label(text='', font_size=16, halign='left', size_hint=(1, 0.1))
        self.layout.add_widget(self.history_text)

        self.recognition_history = []
        self.last_entry_time = datetime.now() - timedelta(seconds=30)  # Initial value for last entry time
        self.recognition_buffer = {}  # Buffer to hold consecutive recognition counts
        self.exit_records = set()  # Track exits to avoid duplicates

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def findEncodings(self, images):
        encodings = []
        for img in images:
            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img)
                if face_locations:
                    encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                    encodings.append(encoding)
                else:
                    print(f"Warning: No face detected in image.")
            except Exception as e:
                print(f"Error processing image: {e}")
        return encodings

    def update_camera(self):
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                self.frame = None

    def toggle_camera(self, instance):
        if self.is_camera_running:
            self.is_camera_running = False
            self.start_button.text = 'Start Camera'
            if self.cap:
                self.cap.release()
        else:
            self.cap = cv2.VideoCapture(cv2.CAP_DSHOW)
            self.is_camera_running = True
            self.start_button.text = 'Stop Camera'
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.start()

    def markAttendance(self, name):
        filename = 'Attendance.csv'
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        entry_time = now.time()

        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                print("Loaded existing attendance CSV:")
                print(df.head())
            else:
                df = pd.DataFrame(columns=['Name', 'Entry', 'EnTime', 'EnDate', 'Unnamed: 4', 'Exit', 'ExTime', 'ExDate'])
                print("Created new attendance DataFrame.")

            date_str = now.strftime('%Y-%m-%d')
            existing_entry = df[(df['Name'] == name) & (df['EnDate'] == date_str)]

            print(f"Current time: {dtString}")
            print(f"Existing entry for {name} on {date_str}: {not existing_entry.empty}")

            # Debugging information
            print(f"Entry time: {entry_time}")
            print(f"Time range for entry: {self.entry_start_time} to {self.entry_stop_time}")
            
            if existing_entry.empty:
                # New entry
                if self.is_within_time_range(entry_time, self.entry_start_time, self.entry_stop_time):
                    new_row = pd.DataFrame([{'Name': name, 'Entry': 'Entry', 'EnTime': dtString, 'EnDate': date_str, 'Unnamed: 4': '', 'Exit': '', 'ExTime': np.nan, 'ExDate': np.nan}])
                    df = pd.concat([df, new_row], ignore_index=True)
                    self.recognition_history.append(f"{dtString} - {name} marked as Entry.")
                    print(f"{dtString} - {name} marked as Entry.")
                else:
                    print(f"{dtString} - {name} attempted entry outside defined time slots.")
            else:
                # Existing entry
                if self.is_within_time_range(entry_time, self.exit_start_time, self.exit_stop_time):
                    if name not in self.exit_records:
                        df.loc[existing_entry.index[0], ['Exit', 'ExTime', 'ExDate']] = ['Exit', dtString, date_str]
                        self.recognition_history.append(f"{dtString} - {name} marked as Exit.")
                        self.exit_records.add(name)
                        print(f"{dtString} - {name} marked as Exit.")
                    else:
                        print(f"{dtString} - {name} exit already recorded.")
                else:
                    print(f"{dtString} - {name} attempted exit outside defined time slots.")

            df.to_csv(filename, index=False)
        except Exception as e:
            print(f"Error marking attendance: {e}")


    def is_within_time_range(self, current_time, start_str, stop_str):
        start_time = datetime.strptime(start_str, '%I:%M %p').time()
        stop_time = datetime.strptime(stop_str, '%I:%M %p').time()
        return start_time <= current_time <= stop_time

    def update(self, dt):
        if self.frame is not None:
            img = self.frame
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    if name in self.recognition_buffer:
                        self.recognition_buffer[name] += 1
                    else:
                        self.recognition_buffer[name] = 1

                    if self.recognition_buffer[name] >= self.RECOGNITION_THRESHOLD:
                        self.markAttendance(name)
                        self.result_label.text = f"Recognition Result: {name}"
                        self.recognition_buffer[name] = 0  # Reset buffer for this name

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            buf = cv2.flip(img, 0).tobytes()
            texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

        if len(self.recognition_history) > 5:
            self.recognition_history.pop(0)

        history_text = "\n".join(self.recognition_history)
        self.history_text.text = history_text

    def open_settings(self, instance):
        settings_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        entry_start_label = Label(text='Entry Start Time (HH:MM AM/PM):', font_size=20)
        self.entry_start_input = TextInput(text=self.entry_start_time, multiline=False, font_size=20)

        entry_stop_label = Label(text='Entry Stop Time (HH:MM AM/PM):', font_size=20)
        self.entry_stop_input = TextInput(text=self.entry_stop_time, multiline=False, font_size=20)

        exit_start_label = Label(text='Exit Start Time (HH:MM AM/PM):', font_size=20)
        self.exit_start_input = TextInput(text=self.exit_start_time, multiline=False, font_size=20)

        exit_stop_label = Label(text='Exit Stop Time (HH:MM AM/PM):', font_size=20)
        self.exit_stop_input = TextInput(text=self.exit_stop_time, multiline=False, font_size=20)

        settings_layout.add_widget(entry_start_label)
        settings_layout.add_widget(self.entry_start_input)
        settings_layout.add_widget(entry_stop_label)
        settings_layout.add_widget(self.entry_stop_input)
        settings_layout.add_widget(exit_start_label)
        settings_layout.add_widget(self.exit_start_input)
        settings_layout.add_widget(exit_stop_label)
        settings_layout.add_widget(self.exit_stop_input)

        save_button = Button(text='Save', on_press=self.save_settings, font_size=20)
        settings_layout.add_widget(save_button)

        settings_popup = Popup(title='Settings', content=settings_layout, size_hint=(0.6, 0.6))
        settings_popup.open()

    def save_settings(self, instance):
        self.entry_start_time = self.entry_start_input.text
        self.entry_stop_time = self.entry_stop_input.text
        self.exit_start_time = self.exit_start_input.text
        self.exit_stop_time = self.exit_stop_input.text
        self.save_time_slots()

    def view_summary(self, instance):
        filename = 'Attendance.csv'
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)

                # Create the main layout for the summary popup
                summary_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

                # Add a label for the summary title
                summary_label = Label(text='Attendance Summary:', font_size=28, bold=True, size_hint=(1, None), height=50)
                summary_layout.add_widget(summary_label)

                # Statistical calculations
                total_entries = len(df[df['Entry'] == 'Entry'])
                total_exits = len(df[df['Exit'] == 'Exit'])
                total_records = len(df)
                total_images = len(self.classNames)  # Assuming self.classNames contains image names

                # Display statistical data in a label
                stats_label = Label(text=f'Total Entries: {total_entries}\nTotal Exits: {total_exits}\nTotal Records: {total_records}\nTotal Images in Database: {total_images}',
                                    font_size=22, halign='left', size_hint=(1, None), height=150)
                summary_layout.add_widget(stats_label)

                # Attendance data display in tabular format
                data_layout = BoxLayout(orientation='vertical', spacing=5, size_hint=(1, None))

                # Add headers
                headers = [col for col in df.columns]
                header_layout = BoxLayout(size_hint=(1, None), height=60)
                for header in headers:
                    header_label = Label(text=header, size_hint=(None, None), height=60, font_size=20, bold=True)
                    header_layout.add_widget(header_label)
                data_layout.add_widget(header_layout)

                # Add data rows
                for index, row in df.iterrows():
                    row_layout = BoxLayout(size_hint=(1, None), height=50, spacing=10)
                    for col in headers:
                        data_label = Label(text=str(row[col]), size_hint=(None, None), height=50, font_size=18)
                        row_layout.add_widget(data_label)
                    data_layout.add_widget(row_layout)

                # Wrap the data layout in a ScrollView for scrolling
                scroll_view = ScrollView(size_hint=(1, 1))
                scroll_view.add_widget(data_layout)
                summary_layout.add_widget(scroll_view)

                # Create and open the summary popup
                summary_popup = Popup(title='Attendance Summary', content=summary_layout, size_hint=(0.9, 0.9))
                summary_popup.open()

            else:
                print(f"No attendance file found at {filename}.")
        except Exception as e:
            print(f"Error viewing summary: {e}")

    def download_attendance(self, instance):
        source_path = 'Attendance.csv'
        # Define a function to handle the download process
        def download_to_selected_folder(selected_folder):
            destination_path = os.path.join(selected_folder, 'Attendance.csv')
            try:
                shutil.copy(source_path, destination_path)
                self.show_notification(f"Attendance file downloaded successfully to {destination_path}")
            except FileNotFoundError:
                self.show_notification(f"Error: File not found at {source_path}")
            except PermissionError:
                self.show_notification(f"Error: Permission denied. Check write access to {destination_path}")
            except Exception as e:
                self.show_notification(f"Error downloading attendance: {e}")

        # Create a Tkinter file dialog for selecting a folder
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        selected_folder = filedialog.askdirectory(title="Select Download Location")
        root.destroy()  # Close the Tkinter window after selection

        if selected_folder:
            download_to_selected_folder(selected_folder)

    def show_notification(self, message):
        notification_layout = ScrollView(size_hint=(None, None), size=(400, 400))
        notification_content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        notification_label = Label(text=message, font_size=20, halign='center', size_hint_y=None)
        notification_label.bind(size=notification_label.setter('text_size'))
        notification_content.add_widget(notification_label)
        notification_layout.add_widget(notification_content)

        notification_popup = Popup(title='Notification', content=notification_layout, size_hint=(0.5, 0.5))
        notification_popup.open()
        pass

if __name__ == '__main__':
    FaceRecognitionApp().run()
