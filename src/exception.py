import sys
from src.logger import logging


# Function extract detailed error information
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # Gives the info in which file or on which line the error has occured or Extarcts exception traceback details
    file_name=exc_tb.tb_frame.f_code.co_filename # Get's the file name where the error occured
    # Formatting the error message with filename, line number, and error description
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name , exc_tb.tb_lineno , str(error)
    )
    
    return error_message # Returns the formatted error message


# Custom Exception class that extends Python's built-in Exception class    
class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message) # Initialize the base Exception class
        
        # Store the formatted error message with additional details
        self.error_message = error_message_detail(error_message,error_detail=error_details)
        
        
    # Override the __str__ method to return a readable error message
    def __str__(self):
        return self.error_message # Returns the detailed error message when printed
    

    
