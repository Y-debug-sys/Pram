import re
import sys

from collections import defaultdict
from .graph_utils import path_to_edge_list
from gurobipy import GRB, Model, GurobiError, quicksum


class LPSolver(object):
    """
    Linear Programming solver for Multi-Commodity Flow (MCF) problems.
    
    This class constructs and solves linear programming formulations for various
    objectives in multi-commodity flow problems, including Maximum Link Utilization (MLU),
    Maximum Total Flow (MTF), and Maximum Concurrent Flow (MCF).
    
    Attributes:
        paths_dict (dict): Dictionary mapping source-destination pairs to paths
        _num_paths (int): Number of paths to consider for each commodity
        _objective (str): Objective function type ('MLU', 'MTF', or 'MCF')
    """

    def __init__(
        self, 
        paths_dict, 
        objective, 
        num_paths 
    ):
        """
        Initialize the LPSolver with paths, objective, and number of paths.
        
        Args:
            paths_dict (dict): Maps (source, destination) tuples to path lists
            objective (str): Optimization objective ('MLU', 'MTF', or 'MCF')
            num_paths (int): Number of paths to consider for each commodity
        """
        super().__init__()
        self.paths_dict = paths_dict
        self._num_paths = num_paths
        self._objective = objective

    @property
    def model(self):
        return self._solver.model

    def _construct_path_lp(self, G, edge_to_paths, num_total_paths, sat_flows):
        """
        Construct the linear program based on paths for multi-commodity flow problem.
        
        Args:
            G (networkx.Graph): Network graph with capacity attributes on edges
            edge_to_paths (dict): Maps each edge to the list of paths that use it
            num_total_paths (int): Total number of paths across all commodities
            sat_flows (list): Saturated flows (not currently used in this implementation)
            
        Returns:
            LinearProgram: A LinearProgram object containing the formulated model
        """
        # Create Gurobi model for MCF problem
        m = Model("MCF Problem Solver")

        # Create variables: one continuous variable for each path representing flow on that path
        path_vars = m.addVars(num_total_paths, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

        # Set objective based on the specified optimization goal
        if self._objective == 'MLU':
            # Minimize Maximum Link Utilization (MLU) objective
            max_link_util_var = m.addVar(
                vtype=GRB.CONTINUOUS, lb=0.0, name="z"
            )

            m.setObjective(max_link_util_var, GRB.MINIMIZE)
            # Add edge utilization constraints: total flow on each edge must not exceed
            # its capacity times the maximum link utilization variable
            for u, v, c_e in G.edges.data("capacity"):
                if (u, v) in edge_to_paths:
                    paths = edge_to_paths[(u, v)]
                    constr_vars = [path_vars[p] for p in paths]
                    m.addConstr(quicksum(constr_vars) <= c_e * max_link_util_var)

            # Add demand equality constraints: total flow for each commodity must equal demand
            commod_id_to_path_inds = {}
            _demand_constrs = []
            for k, d_k, path_ids in self.commodities:
                commod_id_to_path_inds[k] = path_ids
                _demand_constrs.append(
                    m.addConstr(quicksum(path_vars[p] for p in path_ids) == d_k)
                )

        else:
            # Handle MTF (Maximum Total Flow) and MCF (Maximum Concurrent Flow) objectives
            if self._objective == 'MTF':
                # Maximize total flow across all commodities
                obj = quicksum(path_vars)
            elif self._objective == 'MCF':
                # Maximize concurrent flow fraction (alpha) - maximum fraction of demands satisfied
                alpha = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="a")
                m.update()
                # For each commodity, total flow must be at least alpha * demand
                for k, d_k, path_ids in self.commodities:
                    m.addConstr(quicksum(path_vars[p] for p in path_ids) >= alpha * d_k)

                obj = alpha
            
            # Set maximization objective for MTF and MCF
            m.setObjective(obj, GRB.MAXIMIZE)

            # Add edge capacity constraints: total flow on each edge must not exceed capacity
            for u, v, c_e in G.edges.data("capacity"):
                if (u, v) in edge_to_paths:
                    paths = edge_to_paths[(u, v)]
                    constr_vars = [path_vars[p] for p in paths]
                    m.addConstr(quicksum(constr_vars) <= c_e)

            # Add demand constraints: total flow for each commodity must not exceed demand
            commod_id_to_path_inds = {}
            _demand_constrs = []
            for k, d_k, path_ids in self.commodities:
                commod_id_to_path_inds[k] = path_ids
                _demand_constrs.append(
                    m.addConstr(quicksum(path_vars[p] for p in path_ids) <= d_k)
                )

        return LinearProgram(m)
    
    def pre_solve(self, problem=None):
        """
        Preprocess the problem to map paths to commodities and edges.
        
        This method organizes the paths for each commodity and creates mappings
        between edges and the paths that use them. This preprocessing step is
        essential for constructing the path-based linear program formulation.
        
        Args:
            problem (object, optional): Problem instance containing network and commodities
                                       If None, uses self._problem
            
        Returns:
            tuple: A tuple containing (edge_to_paths dict, total number of paths)
        """
        if problem is None:
            problem = self._problem

        # Get the list of commodities (source, destination, demand triplets)
        self.commodity_list = (
            problem.commodity_list
        )
        self.commodities = []
        edge_to_paths = defaultdict(list)  # Maps each edge to the paths using it
        self._path_to_commod = {}          # Maps each path index to its commodity
        self._all_paths = []               # Stores all paths in order

        paths_dict = self.paths_dict
        path_i = 0  # Index for each path in the LP formulation
        for k, (s_k, t_k, d_k) in self.commodity_list:
            paths = paths_dict[(s_k, t_k)]  # Get all paths for this commodity
            path_ids = []
            for path in paths:
                self._all_paths.append(path)

                # For each edge in the path, record that this path uses this edge
                for edge in path_to_edge_list(path):
                    edge_to_paths[edge].append(path_i)
                path_ids.append(path_i)

                # Record which commodity this path belongs to
                self._path_to_commod[path_i] = k
                path_i += 1

            # Store commodity info with its path indices
            self.commodities.append((k, d_k, path_ids))

        return dict(edge_to_paths), path_i

    def _construct_lp(self, sat_flows=[]):
        """
        Construct the linear program by first preprocessing the problem data.
        
        Args:
            sat_flows (list): Saturated flows (not currently used)
            
        Returns:
            LinearProgram: Formulated linear program object
        """
        edge_to_paths, num_paths = self.pre_solve()
        return self._construct_path_lp(
            self._problem.G, edge_to_paths, num_paths, sat_flows
        )

    def solve(self, problem, use_top=False, fixed_total_flows=[], **args):
        """
        Solve the linear program for the given problem instance.
        
        Args:
            problem (object): Problem instance with network graph and commodities
            use_top (bool): Flag to use top-k paths instead of all paths (default False)
            fixed_total_flows (list): Fixed flows to consider (default empty)
            **args: Additional arguments passed to the solver
            
        Returns:
            float: The optimal objective value of the solved LP
        """
        self._problem = problem
        if use_top:
            self._solver = self._construct_lp_top(fixed_total_flows)
        else:
            self._solver = self._construct_lp(fixed_total_flows)
        return self._solver.solve_lp(**args)

    @property
    def sol_dict(self):
        """
        Get solution dictionary mapping commodities to their path flows.
        
        The solution dictionary contains the flow values assigned to each path
        for each commodity in the problem. This provides a detailed breakdown
        of how the demand for each commodity is routed through the network.
        
        Returns:
            dict: Mapping from commodity keys to lists of flow values on paths
        """
        if not hasattr(self, "_sol_dict"):
            sol_dict_def = defaultdict(list)

            for var in self.model.getVars():
                if var.varName.startswith("f["):
                    # Parse the variable name to extract path index
                    match = re.match(r"f\[(\d+)\]", var.varName)
                    p = int(match.group(1))
                    # Map path index to commodity key and get the flow value
                    commod_key = self.commodity_list[self._path_to_commod[p]]
                    sol_dict_def[commod_key].append(var.x)

            self._sol_dict = {}
            sol_dict_def = dict(sol_dict_def)
            for commod_key in self._problem.commodity_list:
                if commod_key in sol_dict_def:
                    self._sol_dict[commod_key] = sol_dict_def[commod_key]
                else:
                    self._sol_dict[commod_key] = []

        return self._sol_dict


class LinearProgram(object):
    """
    Wrapper class for a Gurobi linear program model with configurable parameters.
    
    This class encapsulates a Gurobi model and provides methods to solve it
    with various optimization settings such as tolerances, thread count, etc.
    """
    def __init__(
        self, model, debug_fn=None, DEBUG=False, VERBOSE=False, out=None, gurobi_out=""
    ):
        """
        Initialize the LinearProgram wrapper.
        
        Args:
            model: Gurobi model object to wrap
            debug_fn (callable, optional): Function for debugging variable interpretation
            DEBUG (bool): Enable debug output (default False)
            VERBOSE (bool): Enable verbose output (default False)
            out: Output stream for printing (default stdout)
            gurobi_out (str): Path for Gurobi log file (default "")
        """
        if out is None:
            out = sys.stdout
        self._model = model
        self._debug_fn = debug_fn
        self.DEBUG = DEBUG
        self.VERBOSE = VERBOSE
        self.out = out
        self._gurobi_out = gurobi_out

    def _print(self, *args):
        """Print to the configured output stream."""
        print(*args, file=self.out)

    @property
    def gurobi_out(self):
        """Get the Gurobi log output file path."""
        return self._gurobi_out

    @gurobi_out.setter
    def gurobi_out(self, gurobi_out):
        """
        Set the Gurobi log output file path.
        
        Args:
            gurobi_out (str): Path for Gurobi log file
        """
        if gurobi_out == "stdout" or gurobi_out == "<stdout>":
            self._gurobi_out = "gurobi.log"
        else:
            self._gurobi_out = gurobi_out

    # Note: this is not idempotent: the `model` parameter will be changed after invoking
    # this function
    def solve_lp(
        self, num_threads=None, bar_tol=None, err_tol=None, numeric_focus=False
    ):
        """
        Solve the linear program with specified parameters.
        
        Args:
            num_threads (int, optional): Number of threads for Gurobi solver
            bar_tol (float, optional): Barrier convergence tolerance
            err_tol (float, optional): Optimality and feasibility tolerance
            numeric_focus (bool): Enable numerical focus mode (default False)
            
        Returns:
            float: The optimal objective value, or None if an error occurred
        """
        model = self._model
        # Disable Gurobi's default output since we handle it separately
        model.setParam('OutputFlag', 0)
        if numeric_focus:
            model.setParam("NumericFocus", 1)
        if num_threads:
            model.setParam("Threads", num_threads)
        model.setParam("LogFile", self.gurobi_out)
        try:
            if bar_tol:
                model.Params.BarConvTol = bar_tol
            if err_tol:
                model.Params.OptimalityTol = err_tol
                model.Params.FeasibilityTol = err_tol

            if self.VERBOSE:
                self._print("\nSolving LP")
            model.optimize()

            if self.DEBUG or self.VERBOSE:
                # Print variable values for debugging or verbose output
                for var in model.getVars():
                    # if var.x != 0:
                        if self.DEBUG and self._debug_fn:
                            if not var.varName.startswith("f["):
                                continue
                            u, v, k, s_k, t_k, d_k = self._debug_fn(var)
                            if self.VERBOSE:
                                self._print(
                                    "edge ({}, {}), demand ({}, ({}, {}, {})), flow: {}".format(
                                        u, v, k, s_k, t_k, d_k, var.x
                                    )
                                )
                        elif self.VERBOSE:
                            self._print("{} {}".format(var.varName, var.x))
                if self.VERBOSE:
                    self._print("Obj: %g" % model.objVal)
            return model.objVal
        except GurobiError as e:
            self._print("Error code " + str(e.errno) + ": " + str(e))
        except AttributeError as e:
            self._print(str(e))
            self._print("Encountered an attribute error")

    @property
    def model(self):
        """Get the underlying Gurobi model."""
        return self._model

    @property
    def obj_val(self):
        """Get the optimal objective value of the solved model."""
        return self._model.objVal
