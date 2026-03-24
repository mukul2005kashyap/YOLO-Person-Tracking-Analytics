import time
import cv2  # type: ignore

class AlertSystem:
    def __init__(self, cooldown=3.0):
        self.cooldown = cooldown
        self.last_alert_time = {}  # Map of person_id -> timestamp
        
        # Items that trigger an alert regardless of compliance status
        self.restricted_items = {"camera", "laptop", "bottle", "cell phone", "remote", "keyboard"}

    def process(self, frame, classified_persons, carried_items=None):
        """
        Processes the current frame, checks for alert conditions, 
        and draws alerts directly on the frame.
        """
        current_time = time.time()
        
        # Determine items carried by each person
        person_items = {}
        if carried_items:
            for item in carried_items:
                pid = item.get('associated_person_id')
                if pid is not None:
                    if pid not in person_items:
                        person_items[pid] = []
                    person_items[pid].append(item['class_name'])
                    
        for person in classified_persons:
            pid = person['id']
            status = person['status']
            
            trigger_reasons = []
            
            # Condition 1: Non-Compliant
            if status == "Non-Compliant":
                trigger_reasons.append("Non-Compliant")
                
            # Condition 2: Carrying specific restricted items
            carried_by_person = person_items.get(pid, [])
            restricted_carried = [item for item in carried_by_person if item in self.restricted_items]
            
            if restricted_carried:
                unique_restricted = list(set(restricted_carried))
                trigger_reasons.append(f"Carrying {', '.join(unique_restricted).title()}")
                
            if trigger_reasons:
                # Trigger alert and reset cooldown
                self.last_alert_time[pid] = {
                    "time": current_time,
                    "reason": " | ".join(trigger_reasons)
                }
                
        # Gather all active alerts (those within their cooldown window)
        active_alerts = []
        for pid, data in list(self.last_alert_time.items()):
            if current_time - data["time"] <= self.cooldown:
                active_alerts.append({
                    "id": pid,
                    "reason": data["reason"]
                })
            else:
                # Expired alert
                del self.last_alert_time[pid]
                
        return self.draw_alerts(frame, active_alerts)

    def draw_alerts(self, frame, active_alerts):
        """
        Draws the active alerts in the top-left corner.
        Also draws "ALERT ACTIVE" if there are any alerts.
        """
        if not active_alerts:
            return frame
            
        # Draw "ALERT ACTIVE" at top-center/left
        cv2.putText(frame, "ALERT ACTIVE!!!", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
        y_offset = 80
        for alert in active_alerts:
            text = f"ALERT: ID {alert['id']} {alert['reason']}"
            cv2.putText(frame, text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 35
            
        return frame
