import os 
import pandas as pd

logdir = "/data/pathology/projects/autoaugmentation/from_chansey_review/pba_colab/tmp/ray_results"
logs = os.listdir(logdir)

log_dict = {}
for log in logs:
	if log[-4:] == ".csv":
		try:
			print(log + 'is a csv file, parsing')
			file_path =  os.path.join(logdir,log)
			file = pd.read_csv(file_path)

			min_val_obj_line = file[file.valid_obj == file.valid_obj.min()]#'valid_obj'
			print('min val line '+ str(min_val_obj_line))
			log_dict[log] = str(min_val_obj_line)
		except AttributeError:
			print("File does not have a suitable structure")
	else:
		print(log + 'is not a csv file, skipping')
df = pd.DataFrame.from_dict(log_dict, orient="index")
df.to_csv(os.path.join(logdir,"data.csv"))



