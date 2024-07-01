# Please download the datasets as COCO format

import json
# Load the JSON data
with open(r'G:\lund\master_thesis\DATA_VERSION\Test3.15.v1i.coco\train\_annotations.coco.json', 'r') as file:
    data = json.load(file)

# Loop through each object and delete the array
for annotation in data['annotations']:
    annotation['segmentation'] = []

# Save the modified data back to the file
with open(r'G:\lund\master_thesis\DATA_VERSION\Test3.15.v1i.coco\train\_annotations.coco.json', 'w') as file:
    json.dump(data, file, indent=4)