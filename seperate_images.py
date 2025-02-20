import os
import shutil

# Define dataset directories
dataset_dir = r"C:\Users\babyr\Braintumordetection\archive (1)"

# Define output folders
output_train_yes = os.path.join(dataset_dir, "train_yes")
output_train_no = os.path.join(dataset_dir, "train_no")
output_test_yes = os.path.join(dataset_dir, "test_yes")
output_test_no = os.path.join(dataset_dir, "test_no")

# Create new folders if they don't exist
os.makedirs(output_train_yes, exist_ok=True)
os.makedirs(output_train_no, exist_ok=True)
os.makedirs(output_test_yes, exist_ok=True)
os.makedirs(output_test_no, exist_ok=True)

# List of tumor class folders (in lowercase, adjust to your folder names)
tumor_classes = ["glioma", "meningioma", "pituitary"]  # "yes" is not needed as a class


# Function to move images
def move_images(source_dir, output_yes, output_no):
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):  # Ensure it's a folder
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if class_folder.lower() in tumor_classes:  # Check if it's a tumor class
                    shutil.move(file_path, os.path.join(output_yes, file))
                elif class_folder.lower() == "notumor":  # Handle the "no tumor" case
                    shutil.move(file_path, os.path.join(output_no, file))


# Move images from train and test folders
train_dir = os.path.join(dataset_dir, "Training")
test_dir = os.path.join(dataset_dir, "Testing")

move_images(train_dir, output_train_yes, output_train_no)
move_images(test_dir, output_test_yes, output_test_no)

print("Dataset successfully reorganized!")