import ROOT
import glob
import random
import os
import sys
import array
from tqdm import tqdm

# --- User Configuration ---

# Glob pattern for input ROOT files.
# The '**' allows for recursive searching in subdirectories.
INPUT_GLOB = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2/stop*/**/*.root"

# Directory where the new, mixed ROOT files will be saved.
OUTPUT_DIR = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2_signal_mixed"

# Prefix for the output file names (e.g., "out" -> out_00000.root).
OUTPUT_PREFIX = "out"

# Maximum number of events to write into a single output ROOT file.
# This also defines the size of the in-memory mixing pool.
MAX_EVENTS_PER_FILE_M = 50000

# Number of events to read from each input file in a single pass.
# The user requested 100. This is now a configurable parameter.
EVENTS_PER_CHUNK = 100

# --- Type Mapping for ROOT to Python's array module ---
ROOT_TO_ARRAY_TYPE = {
    'Float_t': 'f', 'Double_t': 'd', 'Int_t': 'i', 'UInt_t': 'I',
    'Short_t': 'h', 'UShort_t': 'H', 'Long64_t': 'q', 'ULong64_t': 'Q',
    'Bool_t': 'B',
}

# --- Core Functions ---

def get_branch_info(tree):
    """
    Analyzes a TTree and extracts information about its branches,
    including name, type, and whether it's a std::vector.
    """
    branch_info = {}
    print("Inspecting branches of the input TTree...")
    for branch in tree.GetListOfBranches():
        name = branch.GetName()
        class_name = branch.GetClassName()
        if 'vector' in class_name:
            try:
                type_str = class_name.split('<')[1].split('>')[0].strip()
                branch_info[name] = {'type': type_str, 'is_vector': True}
            except IndexError:
                print(f"  - WARNING: Could not parse vector type for branch '{name}'. Skipping.")
        else:
            leaf = branch.GetListOfLeaves().At(0)
            if leaf:
                type_name = leaf.GetTypeName()
                branch_info[name] = {'type': type_name, 'is_vector': False}
            else:
                print(f"  - WARNING: Could not get TLeaf for branch '{name}'. Skipping.")
    return branch_info

def write_chunk_to_file(events_chunk, branch_info, output_dir, prefix, file_counter):
    """
    Writes a given chunk of events to a single new ROOT file.
    Returns the updated file counter.
    """
    output_filename = os.path.join(output_dir, f"{prefix}_{file_counter:05d}.root")
    
    f_out = ROOT.TFile(output_filename, "RECREATE")
    t_out = ROOT.TTree("Events", "Mixed and Shuffled Events")
    
    # Create containers (buffers) for each branch
    branch_containers = {}
    for name, info in branch_info.items():
        if info['is_vector']:
            container = ROOT.std.vector(info['type'])()
            branch_containers[name] = container
            t_out.Branch(name, container)
        else:
            array_type_code = ROOT_TO_ARRAY_TYPE.get(info['type'])
            if array_type_code:
                container = array.array(array_type_code, [0])
                branch_containers[name] = container
                t_out.Branch(name, container, f"{name}/{info['type'][0].upper()}")
            else:
                print(f"Warning: Unsupported simple type '{info['type']}' for branch '{name}'.")

    # Fill the new TTree with the data from the chunk
    for event_data in events_chunk:
        for name, container in branch_containers.items():
            value = event_data.get(name)
            if value is None: continue
            
            if branch_info[name]['is_vector']:
                container.clear()
                for item in value:
                    container.push_back(item)
            else:
                container[0] = value
        t_out.Fill()
        
    print(f"Writing {t_out.GetEntries()} events to {output_filename}")
    t_out.Write()
    f_out.Close()
    
    return file_counter + 1

# --- Main Execution ---

def main():
    """Main function to orchestrate the file processing."""
    print("--- ROOT File Mixing Script (Memory-Efficient Version) ---")
    
    # 1. Find all input files
    print(f"Searching for files with pattern: {INPUT_GLOB}")
    file_list = glob.glob(INPUT_GLOB, recursive=True)
    if not file_list:
        print("Error: No files found. Please check INPUT_GLOB.")
        sys.exit(1)
    print(f"Found {len(file_list)} files to process.")
    
    # 2. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        
    # 3. Inspect the first file to determine the TTree structure
    try:
        f_template = ROOT.TFile.Open(file_list[0])
        t_template = f_template.Get("Events")
        branch_info = get_branch_info(t_template)
        f_template.Close()
    except Exception as e:
        print(f"Error inspecting first file '{file_list[0]}': {e}")
        sys.exit(1)

    if not branch_info:
        print("Error: Could not determine branch structure. Exiting.")
        sys.exit(1)

    # --- Memory-Efficient Streaming and Mixing Logic ---
    print("\nStarting memory-efficient streaming and mixing process...")
    
    # Open all files and prepare them as data sources
    input_sources = []
    for file_path in file_list:
        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie(): continue
        t = f.Get("Events")
        if not t:
            f.Close()
            continue
        input_sources.append({'file': f, 'tree': t, 'total': t.GetEntries(), 'current': 0})

    event_pool = []
    file_counter = 0
    total_events_to_process = sum(s['total'] for s in input_sources)
    
    if total_events_to_process == 0:
        print("No events found in any of the input files. Exiting.")
        for source in input_sources: source['file'].Close()
        sys.exit(0)

    with tqdm(total=total_events_to_process, desc="Mixing Events") as pbar:
        while True:
            more_events_to_read = False
            # Iterate through each input file and grab a small chunk of events
            for source in input_sources:
                if source['current'] >= source['total']:
                    continue  # This file is done

                more_events_to_read = True
                
                n_to_read = min(EVENTS_PER_CHUNK, source['total'] - source['current'])
                
                tree = source['tree']
                for i in range(n_to_read):
                    entry_index = source['current'] + i
                    tree.GetEntry(entry_index)
                    
                    event_data = {}
                    for name, info in branch_info.items():
                        branch_val = getattr(tree, name)
                        event_data[name] = list(branch_val) if info['is_vector'] else branch_val
                    event_pool.append(event_data)

                source['current'] += n_to_read
                pbar.update(n_to_read)

            # If the pool is full, shuffle it, write a file, and clear the written part
            if len(event_pool) >= MAX_EVENTS_PER_FILE_M:
                print(f"\nMixing pool is full ({len(event_pool)} events). Shuffling and writing file...")
                random.shuffle(event_pool)
                
                events_to_write = event_pool[:MAX_EVENTS_PER_FILE_M]
                event_pool = event_pool[MAX_EVENTS_PER_FILE_M:]
                
                file_counter = write_chunk_to_file(events_to_write, branch_info, OUTPUT_DIR, OUTPUT_PREFIX, file_counter)

            if not more_events_to_read:
                break

    # Write any remaining events in the pool to a final file
    if event_pool:
        print(f"\nWriting {len(event_pool)} remaining events to the final file...")
        random.shuffle(event_pool)
        write_chunk_to_file(event_pool, branch_info, OUTPUT_DIR, OUTPUT_PREFIX, file_counter)

    # Clean up: close all opened TFiles
    print("\nClosing all input files.")
    for source in input_sources:
        source['file'].Close()
        
    print("\n--- Script finished successfully! ---")

if __name__ == "__main__":
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    main()
