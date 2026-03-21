import os
import pandas as pd
from datetime import datetime

class CSVLogger:
    def __init__(self, filepath="log.csv"):
        self.filepath = filepath
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=["Timestamp", "Person ID", "Event Message"])
            df.to_csv(self.filepath, index=False)
            
        # Keep track of the last logged state for each ID to prevent spamming the CSV
        self.history = {} 

    def log_event(self, person_id, event_message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Only log if the event message is new/changed for this person ID
        if person_id not in self.history or self.history[person_id] != event_message:
            new_row = {"Timestamp": timestamp, "Person ID": f"ID_{person_id}", "Event Message": event_message}
            df = pd.DataFrame([new_row])
            df.to_csv(self.filepath, mode='a', header=False, index=False)
            
            self.history[person_id] = event_message

    def log_detection(self, person_id):
        # A separate initial log to report that the person has just entered the frame
        if person_id not in self.history:
            timestamp = datetime.now().strftime("%H:%M:%S")
            new_row = {"Timestamp": timestamp, "Person ID": f"ID_{person_id}", "Event Message": "Person detected"}
            df = pd.DataFrame([new_row])
            df.to_csv(self.filepath, mode='a', header=False, index=False)
            
            # Note: We do NOT set self.history[person_id] here, 
            # so the subsequent classification status can be logged immediately after.
