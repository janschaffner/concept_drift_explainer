import json
import pandas as pd
from datetime import datetime

class ChangePointParser:
    """
    Parser for CV4CDD change point output
    """
    
    def __init__(self):
        self.change_points = None
        self.change_points_df = None
    
    def load_from_file(self, file_path):
        """
        Load change points from JSON file
        """
        with open(file_path, 'r') as file:
            self.change_points = json.load(file)
        
        # Convert to DataFrame for easier processing
        self.to_dataframe()
        return self.change_points
    
    def load_from_memory(self, json_data):
        """
        Load change points from JSON data in memory
        """
        self.change_points = json_data
        
        # Convert to DataFrame for easier processing
        self.to_dataframe()
        return self.change_points
    
    def to_dataframe(self):
        """
        Convert change points to pandas DataFrame
        
        Note: This method needs to be adapted based on the actual structure
        of the CV4CDD output
        """
        if not self.change_points:
            return None
        
        # This is a placeholder - actual implementation depends on CV4CDD output format
        change_points_list = []
        
        # Assuming change_points has a list of changes with timestamp and attributes
        for cp in self.change_points.get("change_points", []):
            change_points_list.append({
                "timestamp": cp.get("timestamp"),
                "process_id": cp.get("process_id"),
                "change_type": cp.get("change_type"),
                "confidence": cp.get("confidence"),
                "affected_attributes": cp.get("affected_attributes")
            })
        
        self.change_points_df = pd.DataFrame(change_points_list)
        
        # Convert timestamps to datetime objects
        if "timestamp" in self.change_points_df.columns:
            self.change_points_df["timestamp"] = pd.to_datetime(self.change_points_df["timestamp"])
        
        return self.change_points_df
    
    def get_time_range(self):
        """
        Get the time range covered by the change points
        """
        if self.change_points_df is None or "timestamp" not in self.change_points_df.columns:
            return None, None
            
        min_time = self.change_points_df["timestamp"].min()
        max_time = self.change_points_df["timestamp"].max()
        
        return min_time, max_time
    
    def filter_by_timeframe(self, start_date, end_date):
        """
        Filter change points by timeframe
        """
        if self.change_points_df is None:
            return None
            
        mask = (self.change_points_df["timestamp"] >= start_date) &                (self.change_points_df["timestamp"] <= end_date)
               
        return self.change_points_df[mask]
