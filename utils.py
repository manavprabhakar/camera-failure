import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

class Edge():
    def __init__(self,source_node,target_node,edge_degree,edge_stress,impact_vector):
        self.source_node = source_node
        self.target_node = target_node
        self.source = (source_node.x,source_node.y)
        self.target = (target_node.x,target_node.y)
        self.edge_degree = edge_degree + 1
        self.edge_stress = edge_stress
        self.num_frontier_points = 0
        self.impact_vector = impact_vector

class Node():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.stress = 0
        self.visited = False
    
    def update_stress(self,stress):
        self.stress += stress

class StressPropagator():
    def __init__(
            self,
            BREAK_THRESHOLD,
            NN_RADIUS,
            IMPACT_POINT,
            IMPACT_FORCE,
            IMPACT_ANGLE,
            all_points,
            points,
            K = 1,
        ):
        """
        Primary physical and visual values
        BREAK_THRESHOLD: Threshold value beyond which the edge breaks.
        all_points: Dictionary of Nodes stored with their coordinate value as the key.
        all_edges: Dictionary where edges are added based on their degree (for visualization purposes).
        tree: KDTree storing all points for easy navigation.
        points: Coordinates of all points in the mesh.
        """
        self.BREAK_THRESHOLD = BREAK_THRESHOLD
        self.all_points = all_points
        # Build KD-Tree for the points
        self.tree = KDTree(points)
        self.points = points
        
        self.K = K
        self.all_edges = {}
        self.NN_RADIUS = NN_RADIUS
        IMPACT_NODE = Node(
            x = IMPACT_POINT[0][0],
            y = IMPACT_POINT[0][1]
        )
        # Adding stress values at each nodes.
        IMPACT_NODE.stress = IMPACT_FORCE

        INIT_IMPACT_ANGLE = np.deg2rad(IMPACT_ANGLE)
        impact_vector = np.array([[np.cos(INIT_IMPACT_ANGLE),np.sin(INIT_IMPACT_ANGLE)]]).T

        INIT_EDGE = Edge(
            source_node  = IMPACT_NODE,
            target_node  = IMPACT_NODE,
            edge_degree  = -1,
            edge_stress  = IMPACT_FORCE,
            impact_vector = impact_vector

        )

        self.initiate_propagation(
            impact_point = (IMPACT_POINT[0][0],IMPACT_POINT[0][1]),
            impact_force = IMPACT_FORCE,
            impact_vector = impact_vector,
            init_edge = INIT_EDGE,
        )

    def initiate_propagation(self,impact_point,impact_vector,impact_force,init_edge):
        
        # Converting to a 2D 
        impact_point = np.array([impact_point])
        indices = self.tree.query_radius(impact_point,r = self.NN_RADIUS)
        frontier_points = self.points[indices[0]]
        # Removing the impact point from the nearest neighbors
        frontier_points = frontier_points[np.sum(frontier_points!=impact_point,axis=1)!=0]
        
        # Create a vector of all NNs to find the direction between the NN and impact vector
        NN_vectors = frontier_points - impact_point # Shape -> (N,2)
        # Converting NN_vectors to unit vectors and broadcasting for efficient division
        NN_vectors = NN_vectors/np.linalg.norm(NN_vectors,axis=1)[:,None]
        # NN_vectors is the new impact vector for aall frontier_points
        
        cos_angles = np.dot(NN_vectors,impact_vector)[:,0]
        # Clipping to cosine ranges to avoid overflows due to numerical erros.
        cos_angles = np.clip(cos_angles, -1, 1)
                
        # Now computing stress along each direction.
        stress = np.abs(impact_force*cos_angles)

        self.initial_impact_NN_vectors = NN_vectors
        self.initial_impact_force = stress


        for i in range(NN_vectors.shape[0]):
            child_edge = Edge(
                source_node = self.all_points[tuple(impact_point[0])],
                target_node = self.all_points[(frontier_points[i][0],frontier_points[i][1])],
                edge_degree = init_edge.edge_degree,
                edge_stress = stress[i],
                impact_vector = NN_vectors[i],
            )
            if child_edge.edge_degree not in self.all_edges:
                self.all_edges[child_edge.edge_degree] = []
                
            self.all_edges[child_edge.edge_degree].append(child_edge)
            self.propagate(
                impact_point = (frontier_points[i][0],frontier_points[i][1]),
                impact_force = stress[i],
                impact_vector = NN_vectors[i][:,None],
                parent_edge = init_edge,
            )
        
    def propagate(self,impact_point,impact_force,parent_edge,impact_vector):
        """
        impact_point (x,y): The current point of impact, force traverses through this point.
        impact_vector (i,j): Unit vector signifying direction of the impact force.
        impact_force (N): The force at current impact point.
        parent_edge  (Edge): The edge from the parent node. Required for propagation

        """
        
        # Converting to a 2D 
        impact_point = np.array([impact_point])
        # Query the KD-Tree for nearest neigbors within a given radius
        indices = self.tree.query_radius(impact_point, r = self.NN_RADIUS)
        frontier_points = self.points[indices[0]]
        
        # Removing the impact point from the nearest neighbors
        frontier_points = frontier_points[np.sum(frontier_points!=impact_point,axis=1)!=0]
        
        # Create a vector of all NNs to find the direction between the NN and impact vector
        NN_vectors = frontier_points - impact_point # Shape -> (N,2)
        # Converting NN_vectors to unit vectors and broadcasting for efficient division
        NN_vectors = NN_vectors/np.linalg.norm(NN_vectors,axis=1)[:,None]
        
        # Impact vector magnitude will be 1 (since it's a unit vector).
        # Adding zero index to avoid broadcasting
        cos_angles = np.dot(NN_vectors,impact_vector)[:,0]
        # Clipping to cosine ranges to avoid overflows due to numerical erros.
        cos_angles = np.clip(cos_angles, -1, 1)
        angles = np.arccos(cos_angles)

        # Now computing stress along each direction.
        # Smaller the angle, higher the impact force.
        stress = impact_force*cos_angles
        

        for i,pt in enumerate(frontier_points):
            f_x,f_y = pt[0],pt[1]
            self.all_points[(f_x,f_y)].update_stress(stress[i])

        propagatory_idx = np.argsort(stress)[-self.K:]
        stress = stress[propagatory_idx]
        points_to_propagate = frontier_points[propagatory_idx]
        angles = angles[propagatory_idx]
        next_impact_vector = NN_vectors[propagatory_idx]
        

        # Propagating along points which have maximum stress
        points_to_propagate = points_to_propagate[stress>self.BREAK_THRESHOLD]
        angles = angles[stress>self.BREAK_THRESHOLD]
        next_impact_vector = next_impact_vector[stress>self.BREAK_THRESHOLD]
        stress = stress[stress>self.BREAK_THRESHOLD]
        
        # ? Do we want to visit all these frontiers? I suspect no, then we need to figure out 
        # one frontier/ k frontiers (k<n)
        if points_to_propagate.shape[0]>0:
            for i,pt in enumerate(points_to_propagate):
                # The new point is a frontier only if it has not bee visited already.
                # This takes out all uncertainties, more physics-based because node visit is because of angle and magnitude of force.
                if not self.all_points[tuple(pt)].visited:
                    # Create Edges between the impact point and points to propagate
                    
                    child_edge = Edge(
                        source_node = self.all_points[tuple(impact_point[0])],
                        target_node = self.all_points[tuple(pt)],
                        edge_degree = parent_edge.edge_degree,
                        edge_stress = stress[i],
                        impact_vector = NN_vectors[i],
                    )

                    if child_edge.edge_degree not in self.all_edges:
                        self.all_edges[child_edge.edge_degree] = []
                
                    self.all_edges[child_edge.edge_degree].append(child_edge)
                    # Setting the target node to be visited because we are going 
                    # there in this iteration.
                    self.all_points[tuple(pt)].visited = True
                    self.propagate(
                        impact_point = pt,
                        impact_force = stress[i],
                        parent_edge = child_edge,
                        impact_vector = next_impact_vector.T,
                    )