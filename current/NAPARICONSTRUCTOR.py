import numpy as np
import napari

FILENAME = 'obstaclemap.npy'

# 1. Open the viewer and edit your array
my_array = np.ones((220, 110), dtype=np.int8)
# my_array = np.load(FILENAME)

viewer = napari.Viewer()
labels_layer = viewer.add_labels(my_array, name='My Paint Layer')

labels_layer.color = {
        2: 'red',           #OBSTACLE
        1: 'white', #blue #FREE SPACE
    }

# When painting, use label 2, red
labels_layer.selected_label = 2

# Execution pauses here until you close the napari window
napari.run()

# 2. Extract the modified array from the layer
edited_array = labels_layer.data

edited_array[edited_array == 1] = 0
edited_array[edited_array == 2] = 1

# 3. Save it to disk
np.save(FILENAME, edited_array)

print("Array successfully saved!")