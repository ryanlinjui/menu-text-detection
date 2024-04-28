import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def vertex2text(text_vertex:dict, is_patches_return:bool=False):
    patches_list = []
    return_patches = []
    raw_text = "" 
    SCALE = 0.8 # Border scale rate for creating scaled points
    scale_factor = 1 - SCALE

    data = text_vertex["data"].copy()
    x_max, y_max = data[0]["vertex"]["br"]
    data.pop(0)

    for item in data:
        points = [item["vertex"]["tl"], item["vertex"]["tr"], item["vertex"]["br"], item["vertex"]["bl"]]

        # Calculate scaled points based on the scale factor
        scaled_points = [
            [points[0][0] + (points[1][0] - points[0][0]) * scale_factor, points[0][1] + (points[3][1] - points[0][1]) * scale_factor],
            [points[1][0] + (points[0][0] - points[1][0]) * scale_factor, points[1][1] + (points[2][1] - points[1][1]) * scale_factor],
            [points[2][0] + (points[3][0] - points[2][0]) * scale_factor, points[2][1] + (points[1][1] - points[2][1]) * scale_factor],
            [points[3][0] + (points[2][0] - points[3][0]) * scale_factor, points[3][1] + (points[0][1] - points[3][1]) * scale_factor]
        ]

        scaled_points.append(scaled_points[0]) # closed shape for patch
        path = Path(scaled_points)    
        patch = patches.PathPatch(path, facecolor="none", edgecolor="black", lw=1)
        patches_list.append(patch)

    last_patch = [None]
    
    # Iterate over y-axis positions
    for x_line in range(0, y_max):
        flag = True
        
        # Check intersection of patches with a horizontal line
        for i in range(len(patches_list)):
            if patches_list[i].get_path().intersects_path(Path([(0, x_line), (x_max, x_line)])):
                flag = False
                # If last_patch is None, update it with the current data item
                if last_patch[0] == None:
                    last_patch[0] = {
                        "data": data[i],
                        "patch": patches_list[i]
                    }
                    continue
                
                # If data[i] is not in last_patch, append it
                if any(data[i] == d["data"] for d in last_patch) == False:
                    last_patch.append(
                        {
                            "data": data[i],
                            "patch": patches_list[i]
                        }
                    )

        # If there is no intersection and last_patch is not None
        if (flag or x_line == y_max - 1) and last_patch != [None]:
            # Sort data based on the x-coordinate of the top-left vertex
            sorted_data = sorted(last_patch, key=lambda x: (x["data"]["vertex"]["tl"][0]))
            
            # Concatenate the text from sorted_data to raw_text
            for d in sorted_data:
                raw_text += d["data"]["text"]
                return_patches.append(d["patch"])
            
            # Add a newline at the end of the line
            raw_text += "\n"
            
            # Reset last_patch
            last_patch = [None]
    if is_patches_return:
        return raw_text, return_patches
    else:
        return raw_text