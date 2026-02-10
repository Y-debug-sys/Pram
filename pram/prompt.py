import os
from PIL import Image


class PNGLoader:
    """
    A class for loading and managing PNG images from a specified folder.
    
    This class loads all PNG files from a given folder path, stores them in memory,
    and provides methods to retrieve individual images or all images as a list.
    
    Args:
        folder_path (str): Path to the directory containing PNG files to load
    """
    def __init__(self, folder_path):
        # Dictionary to store all loaded images with filename as key
        self.images = {}
        self.num_images = 0
        self.all_imgs = None
        
        # Iterate through the folder and load all PNG files
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".png"):
                full_path = os.path.join(folder_path, fname)
                # Convert all images to RGB mode for consistency
                self.images[fname] = Image.open(full_path).convert("RGB")
                self.num_images += 1

    def get(self, fname):
        """
        Return the PIL.Image object for a given filename.
        
        Args:
            fname (str): Filename of the image to retrieve
            
        Returns:
            PIL.Image or None: The image object if found, None otherwise
        """
        return self.images.get(fname, None)
    
    def get_all(self, prefix='source', W=128, H=128):
        """
        Retrieve all images as a list, resizing them to specified dimensions.
        
        Args:
            prefix (str): Prefix for image filenames (default 'source')
            W (int): Width to resize images to (default 128)
            H (int): Height to resize images to (default 128)
            
        Returns:
            list: List of resized PIL.Image objects
        """
        if self.all_imgs is not None:
            return self.all_imgs
        else:
            self.all_imgs = []
        
        for idx in range(self.num_images):
            # print(f'{prefix}_{idx}')
            img = self.get(f'{prefix}_{idx}.png')
            self.all_imgs.append(img.resize((W, H)))
        
        return self.all_imgs


def prepare_prompt_by_source(
    last_demands, 
    num_nodes, 
    num_paths=4, 
    obj_name='MLU',
    history_mean=None,
    same_cap=True
):
    """
    Prepare prompts for multi-commodity flow planning focusing on individual source nodes.
    
    This function creates a prompt for each source node in the network, asking the model
    to determine path weights for routing flows from that source to all destination nodes.
    
    Args:
        last_demands (torch.Tensor): Recent demand values between node pairs
        num_nodes (int): Total number of nodes in the network
        num_paths (int): Number of candidate paths between each node pair (default 4)
        obj_name (str): Name of the objective function ('MLU', 'MTF', or 'MCF')
        history_mean (torch.Tensor, optional): Historical average demands
        same_cap (bool): Whether all links have the same capacity (default True)
        
    Returns:
        list: List of prompts, one for each source node
    """
    objective = obj_name.upper()
    cap_dcp = "capacity-1-" if same_cap else ""
    role = "<|im_start|>system\nYou are an expert in multi-commodity flow planning. <|im_end|>\n"
    task = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> Based on the all traversed {cap_dcp}links from n (red) to others in the image and other info, determine path weights for each pair. "
    
    if objective == 'MLU':
        obj_dcp = "## Goal: \nThe objective is to minimize max link utilization--the maximum utilization of any link in the network. "
    elif objective == 'MTF': 
        obj_dcp = "## Goal: \nThe objective is to maximize total flow--the total flow of all commodities in the network. "
    elif objective == 'MCF':
        obj_dcp = "## Goal: \nThe objective is to maximize concurrent flow--the maximum parallel flow of all commodities in the network. "
    else:
        raise NotImplementedError()

    history_dcp = f"## Description: \nThere are {str(num_nodes)} nodes in the topology and {str(num_paths)} shortest paths between each node pair are used as condidate paths. No allocation is needed for a node to itself. "
    
    output_dcp = "## Format: \n Make sure to follow the answer template. [<source-0-path1-weight>, ..., <source-0-path4-weight>], [<source-1-...], ...; The sum of the weights should equal 1.  \n"
        
    last_demands = last_demands.reshape(num_nodes, -1)
    prompts = []
    for i in range(num_nodes):
        
        if history_mean is None:
            demand_dcp = f"## Demands: \nThe most recent demands from (source, 0) to (source, {str(num_nodes-1)}) are as follows. \n"
            demands = f"{str(last_demands[i, :].cpu().numpy())}. "
        else:
            demand_dcp = f"## Demands: \nThe most recent and history mean demands from (source, 0) to (source, {str(num_nodes-1)}) are as follows. \n"
            demands = f"{str(last_demands[i, :].cpu().numpy())}; \n{str(history_mean.reshape(num_nodes, -1)[i, :].cpu().numpy())}. "

        prompt = (
            f"{role} {task} Current source node is {str(i)}. \n\n"
            f"{obj_dcp} \n\n"
            f"{history_dcp} \n\n"
            f"{demand_dcp} {demands} \n\n"
            f"{output_dcp}ASSISTANT"
        )
        prompts.append(prompt)

    return prompts


def prepare_prompt_by_group(
    last_demands,
    num_nodes,
    num_agents,
    num_paths=4,
    obj_name='MLU',
    history_mean=None,
    same_cap=True
):
    """
    Prepare prompts grouped by agents for multi-commodity flow planning.
    
    Each agent handles multiple source nodes sequentially. This function divides
    the source nodes among agents and creates a prompt for each agent that
    covers all its assigned source nodes.
    
    Args:
        last_demands (torch.Tensor): Recent demand values between node pairs
        num_nodes (int): Total number of nodes in the network
        num_agents (int): Number of agents to distribute the work
        num_paths (int): Number of candidate paths between each node pair (default 4)
        obj_name (str): Name of the objective function ('MLU', 'MTF', or 'MCF')
        history_mean (torch.Tensor, optional): Historical average demands
        same_cap (bool): Whether all links have the same capacity (default True)
        
    Returns:
        list: List of prompts, one for each agent
    """
    # Validate that num_agents is within valid range
    assert num_agents > 0 and num_agents <= num_nodes

    objective = obj_name.upper()
    cap_dcp = "capacity-1-" if same_cap else ""

    role = (
        "SYSTEM\n"
        "You are an expert in multi-commodity flow planning."
        "\n"
    )

    task = (
        "USER\n"
        "\n\n\n <|vision_start|><|image_pad|><|vision_end|>"
        f"Based on the all traversed {cap_dcp}links from the highlighted source nodes to others in the image and other info, "
        "determine path weights for each source-destination pair. "
    )

    if objective == 'MLU':
        obj_dcp = (
            "## Goal:\n"
            "The objective is to minimize max link utilization "
            "-- the maximum utilization of any link in the network."
        )
    elif objective == 'MTF':
        obj_dcp = (
            "## Goal:\n"
            "The objective is to maximize total flow "
            "-- the total flow of all commodities in the network."
        )
    elif objective == 'MCF':
        obj_dcp = (
            "## Goal:\n"
            "The objective is to maximize concurrent flow "
            "-- the maximum parallel flow of all commodities in the network."
        )
    else:
        raise NotImplementedError()

    history_dcp = (
        "## Description:\n"
        f"There are {num_nodes} nodes in the topology and "
        f"{num_paths} shortest paths between each node pair are used as candidate paths. "
        "No allocation is needed for a node to itself."
    )

    output_dcp = (
        "## Format:\n"
        "Make sure to follow the answer template.\n"
        "[<source-0-path1-weight>, ..., <source-0-path4-weight>], "
        "[<source-1-...>], ...;\n"
        "The sum of the weights for each source should equal 1."
        "\n"
    )

    # ---------- reshape demands ----------
    last_demands = last_demands.reshape(num_nodes, -1)
    if history_mean is not None:
        history_mean = history_mean.reshape(num_nodes, -1)

    # ---------- sequential grouping ----------
    sources = list(range(num_nodes))
    chunk_size = (num_nodes + num_agents - 1) // num_agents
    source_groups = [
        sources[i:i + chunk_size]
        for i in range(0, num_nodes, chunk_size)
    ]

    prompts = []

    # ---------- build prompt per agent ----------
    for agent_id, group_sources in enumerate(source_groups):

        demand_dcp = (
            "## Demands:\n"
            f"This agent is responsible for source nodes {group_sources}. "
            "The most recent demands for each source (to all destinations) are:\n"
        )

        demand_lines = []
        for s in group_sources:
            if history_mean is None:
                demand_lines.append(
                    f"- Source {s}: {last_demands[s, :].cpu().numpy()}"
                )
            else:
                demand_lines.append(
                    f"- Source {s}: {last_demands[s, :].cpu().numpy()}; "
                    f"History mean: {history_mean[s, :].cpu().numpy()}"
                )

        demands = "\n".join(demand_lines)

        prompt = (
            f"{role}"
            f"{task}"
            f"Current agent index is {agent_id}. "
            f"It controls source nodes {group_sources}.\n\n"
            f"{obj_dcp}\n\n"
            f"{history_dcp}\n\n"
            f"{demand_dcp}{demands}\n\n"
            f"{output_dcp}"
            "ASSISTANT"
        )

        prompts.append(prompt)

    return prompts


# Example usage
if __name__ == "__main__":
    loader = PNGLoader("./Pram/prompt/GEANT_figs")   # Replace with your folder path
    # img = loader.get("source_0.png")         # Input filename
    all_imgs = loader.get_all()
    # print(loader.num_images)
    img = all_imgs[0]
    if img:
        img.show()                          # Display the image
    else:
        print("File not found!")