"""
A recording module for save results with configs
"""
import os
import sys
import time
import shutil
import datetime

class Experiment_Recoding():
    def __init__(self, target_folder, message, backup_list = ['config.py', 'model.py'], backup_file_path = None):
        self.target_folder = target_folder # result folder
        self.msg = message # write message
        self.backup_lst = backup_list
        self.backup_file_path = backup_file_path
        
        # --- run --- #
        self._check_and_create_folder()
        self._write_msg_first()
        self._copy_file()
        
    def _check_and_create_folder(self):
        if os.path.exists(self.target_folder):
            print("%s has already exist, overwrite it" % self.target_folder)
        else:
            os.makedirs(self.target_folder)
            print("Creating folder: %s" % (self.target_folder))
    
    def _copy_file(self):
        if self.backup_file_path is None:
            self.backup_file_path = os.getcwd()
        
        for i in self.backup_lst:
            source_file = os.path.join(self.backup_file_path, i)
            target_file = os.path.join(self.target_folder, i)
            print("copying %s to %s" % (source_file, target_file))
            shutil.copyfile(source_file, target_file)
    
    def _write_msg_first(self):
        self.rec_file = os.path.join(self.target_folder, "msg.txt")
        current_time = str(datetime.datetime.now())
        with open(self.rec_file, "w") as f:
            f.write("Recording time: %s \n" % (current_time))
            f.write(self.msg + '\n')
        print("Write recording file to %s" % (self.rec_file))
    
    def write_new_msg(self, msg):
        with open(self.rec_file, 'a') as f:
            f.write(msg + '\n')
            