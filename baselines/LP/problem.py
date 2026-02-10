import numpy as np


def commodity_gen(mat, with_val=True, skip_zero=True):
    """
    Generator function to iterate over commodities in a traffic matrix.
    
    Iterates through all source-destination pairs in the traffic matrix,
    skipping diagonal elements (self-loops) and optionally zero values.
    
    Args:
        mat (numpy.ndarray): 2D traffic matrix where mat[i][j] represents
                            demand from node i to node j
        with_val (bool): If True, yields (src, dst, demand) tuples;
                        If False, yields (src, dst) tuples only
        skip_zero (bool): If True, skips entries with zero demand
        
    Yields:
        tuple: Either (src, dst, demand) if with_val=True or (src, dst) if with_val=False
    """
    for x in range(mat.shape[0]):
        for y in range(mat.shape[-1]):
            # always skip diagonal values (no self-demands)
            if x == y:
                continue
            if skip_zero and mat[x, y] == 0: 
                continue
            if with_val:
                yield x, y, mat[x, y]
            else:
                yield x, y



class Problem(object):
    """
    Represents a network optimization problem with graph structure and traffic demands.
    
    This class encapsulates a network flow problem with a graph structure and a traffic
    matrix defining demands between node pairs. It provides methods to access various
    properties of the problem and utilities for working with the network structure.
    
    Attributes:
        G (networkx.Graph): Network graph with capacity attributes on edges
        traffic_matrix (numpy.ndarray): Matrix of demands between node pairs
        capacity_seed (int): Seed for random capacity generation
    """

    def __init__(
        self,
        G,
        traffic_matrix=None,
        seed=0,
        **kwargs
    ):
        """
        Initialize the network optimization problem.
        
        Args:
            G (networkx.Graph): Network graph with capacity attributes on edges
            traffic_matrix (numpy.ndarray, optional): Matrix of demands between node pairs
            seed (int): Seed for random capacity generation (default 0)
            **kwargs: Additional keyword arguments
        """
        self.G = G
        self.capacity_seed = seed
        self.traffic_matrix = traffic_matrix

    ###########################
    # Public instance methods #
    ###########################
    def print_stats(self):
        """
        Print basic statistics about the problem instance.
        
        Outputs the number of nodes, edges, and commodities in the network.
        """
        print("Num nodes: ", len(self.G.nodes))
        print("Num edges: ", len(self.G.edges))
        print("Num commodities: ", len(self.commodity_list))

    def copy(self):
        """
        Create a deep copy of the problem instance.
        
        Returns:
            Problem: A new Problem instance with copied graph and traffic matrix
        """
        G = self.G.copy()
        traffic_matrix = self.traffic_matrix.copy()
        problem = Problem(G, traffic_matrix)
        # problem.name = self.name
        problem.capacity_seed = self.capacity_seed
        return problem

    ##############
    # Properties #
    ##############
    @property
    def G(self):
        """
        Get the network graph.
        
        Returns:
            networkx.Graph: The network graph with capacity attributes on edges
        """
        return self._G

    @G.setter
    def G(self, G):
        """
        Set the network graph.
        
        Args:
            G (networkx.Graph): Network graph to assign
        """
        self._G = G

    @property
    def traffic_matrix(self):
        """
        Get the traffic demand matrix.
        
        Returns:
            numpy.ndarray: Matrix of demands between node pairs
        """
        return self._traffic_matrix

    @traffic_matrix.setter
    def traffic_matrix(self, traffic_matrix):
        """
        Set the traffic demand matrix and invalidate cached commodity list.
        
        Args:
            traffic_matrix (numpy.ndarray): Matrix of demands between node pairs
        """
        self._traffic_matrix = traffic_matrix
        # invalidate commodity list attributes, since we've updated the traffic
        # matrix
        # self._invalidate_commodity_lists()

    @property
    def capacity_seed(self):
        """
        Get the capacity seed.
        
        Returns:
            int: The seed value used for capacity-related randomness
        """
        return self._capacity_seed

    @capacity_seed.setter
    def capacity_seed(self, capacity_seed):
        """
        Set the capacity seed.
        
        Args:
            capacity_seed (int): Seed value for capacity-related randomness
        """
        self._capacity_seed = capacity_seed

    # set both traffic seed and capacity seed to the same seed; otherwise
    # invoke the normal setter function
    def __setattr__(self, name, value):
        """
        Custom attribute setting that handles the 'seed' property specially.
        
        When 'seed' is set, it assigns the value to both traffic and capacity seeds.
        
        Args:
            name (str): Name of the attribute to set
            value (any): Value to assign to the attribute
        """
        if name == "seed":
            super().__setattr__("_capacity_seed", value)
        else:
            super().__setattr__(name, value)

    @property
    def edges_list(self):
        """
        Get the list of edges in the network.
        
        Caches the result after the first computation for efficiency.
        
        Returns:
            list: List of edges in the network
        """
        if not hasattr(self, "_edges_list"):
            self._edges_list = list(self.G.edges)
        return self._edges_list

    @property
    def commodity_list(self):
        """
        Get the list of commodities (non-zero demands) in the traffic matrix.
        
        Creates a list of commodities represented as (index, (src, dst, demand)) tuples
        where demand is non-zero. Caches the result after the first computation.
        
        Returns:
            list: List of commodities with non-zero demands
        """
        # print(self.traffic_matrix)
        if not hasattr(self, "_commodity_list"):
            self._commodity_list = list(
                enumerate(commodity_gen(self.traffic_matrix))
            )
        return self._commodity_list
    
    @property
    def all_commodity_list(self):
        """
        Get the list of all commodities in the traffic matrix (including zero demands).
        
        Creates a list of all commodities represented as (index, (src, dst, demand)) tuples
        including those with zero demands. Caches the result after the first computation.
        
        Returns:
            list: List of all commodities, including those with zero demands
        """
        # print(self.traffic_matrix)
        if not hasattr(self, "_all_commodity_list"):
            self._all_commodity_list = list(
                enumerate(commodity_gen(self.traffic_matrix, skip_zero=False))
            )
        return self._all_commodity_list

    @property
    def edge_idx(self):
        """
        Get a mapping from edges to their indices in the network.
        
        Returns:
            dict: Dictionary mapping each edge tuple to its index position
        """
        return {edge: e for e, edge in enumerate(self.G.edges)}

    @property
    def is_traffic_matrix_full(self):
        """
        Check if the traffic matrix has demands for all possible node pairs.
        
        Returns True if every non-diagonal entry in the traffic matrix has a non-zero demand.
        
        Returns:
            bool: True if all possible commodities have demands, False otherwise
        """
        return (
            len(self.commodity_list)
            == self.traffic_matrix.size - self.traffic_matrix.shape[0]
        )

    @property
    def total_demand(self):
        """
        Calculate the total demand across all commodities.
        
        Returns:
            float: Sum of all demands in the traffic matrix
        """
        return np.sum(self.traffic_matrix)

    @property
    def total_capacity(self):
        """
        Calculate the total capacity across all edges in the network.
        
        Returns:
            float: Sum of all edge capacities in the network
        """
        return sum(cap for _, _, cap in self.G.edges.data("capacity"))

    # ###########################
    # # Abstract method (sorta) #
    # ###########################
    # @property
    # def name(self):
    #     if hasattr(self, "_name"):
    #         return self._name
    #     raise NotImplementedError(
    #         "name needs to be implemented in the subclass: {}".format(self.__class__)
    #     )

    # @name.setter
    # def name(self, name='mcf_problem'):
    #     self._name = name
