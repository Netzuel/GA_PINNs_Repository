# Import modules
import os
import shutil

if "Images" in os.listdir("."):
	shutil.rmtree("Images/")
if "Models_Data" in os.listdir("."):
	shutil.rmtree("Models_Data/")

if "Images" not in os.listdir("."):
	os.mkdir("Images/")
if "Models_Data" not in os.listdir("."):
	os.mkdir("Models_Data/")
	os.mkdir("Models_Data/Model_Saved")


















































