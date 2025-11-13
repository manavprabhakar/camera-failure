import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_delauney(delaunay_graph,points,fig_height=50,fig_width=50,save=True,visualize=False):    

    plt.figure(figsize=(fig_height,fig_width))
    plt.triplot(points[:,0], points[:,1], delaunay_graph.simplices)
    plt.plot(points[:,0], points[:,1], 'o')
    if save:
        plt.savefig('delaunay_output')
    if visualize:
        plt.show()
        
def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def direction_comparison(sun_direction, point1, point2):
    # Calculate the direction vector for the line
    line_direction = np.array(point2) - np.array(point1)
    
    # Normalize the direction vector of the line
    line_direction_unit = unit_vector(line_direction)
    
    # Calculate the dot product
    dot_product = np.dot(sun_direction, line_direction_unit)
    
    # Calculate the angle in radians
    angle = np.arccos(dot_product)
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle)
    
    return angle_degrees, dot_product

def generate_glass_image(all_edges,H=384, W=1280, min_thickness=1, max_thickness=4,pts_img = None,BREAK_THRESHOLD=300,sun_angle=90):
    sun_angle_rad = np.radians(sun_angle)
    sun_vector = np.array([np.cos(sun_angle_rad),np.sin(sun_angle_rad)])

    # broken_glass_img = np.copy(pts_img)
    broken_glass_img = np.zeros((H,W,3), dtype=np.uint8)
    max_stress = max(edge.edge_stress for edge_list in all_edges.values() for edge in edge_list if edge.edge_degree > 1)
    
    prev_sp = (None,None)
    for edge_list in all_edges.values():
        for edge in edge_list:
            
            # Normalize the stress value for the current edge
            normalized_stress = edge.edge_stress / max_stress
            # Calculate thickness using a non-linear transformation (square root in this case)
            # Non-linear scaling can make the visual decay more gradual
            thickness = min_thickness + (np.sqrt(normalized_stress/10) * (max_thickness - min_thickness))
            thickness = int(thickness)
            
            if edge.edge_degree == 1:
                # Join them with the previous edge
                cur_sp = (edge.source[1],edge.source[0])
                if prev_sp[0]!=None:
                    angle,_ = direction_comparison(sun_direction=sun_vector,point1=start_point,point2=end_point)
                    # thickness = (int)(max(1,3*(edge.edge_stress/max_stress)))
                    grey_value = max(0,200*(1 - abs(np.cos(np.radians(angle)))))
                    cv2.line(broken_glass_img, prev_sp, cur_sp, line_color, thickness=1)
                prev_sp = cur_sp
                # continue
            
            start_point = (edge.source[1], edge.source[0])
            end_point = (edge.target[1], edge.target[0])

            # Draw the line on the image using cv2.line
            angle,_ = direction_comparison(sun_direction=sun_vector,point1=start_point,point2=end_point)
            
            # grey_value = max(0,200*(angle/90))
            grey_value = max(0,200*(1 - abs(np.cos(np.radians(angle)))))
            # thickness = (int)(max(1,3*(edge.edge_stress/max_stress)))
            line_color = (grey_value, grey_value, grey_value)
            cv2.line(broken_glass_img, start_point, end_point, line_color, thickness=1)
        # broken_glass_img = connect_radial(broken_glass_img=broken_glass_img,all_edges=all_edges)
    return broken_glass_img