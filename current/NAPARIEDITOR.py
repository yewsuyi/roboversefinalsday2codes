import numpy as np
import napari

# FILENAME = 'obstaclemap.npy' #TODO SWITCH
# FILENAME = 'scanmap.npy' #TODO SWITCH
FILENAME = 'bousphedron.npy'

# 1. Open the viewer and edit your array
# my_array = np.zeros((512, 512), dtype=np.int32)
my_array = np.load(FILENAME)

viewer = napari.Viewer()
labels_layer = viewer.add_labels(my_array, name='My Paint Layer')

# Execution pauses here until you close the napari window
napari.run()

# 2. Extract the modified array from the layer
edited_array = labels_layer.data

# 3. Save it to disk
np.save(FILENAME, edited_array)

print("Array successfully saved!")