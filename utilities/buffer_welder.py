import pickle as pkl
import os 

get_out = "demos/"
pickle_name = "/demos_finished.pkl"
demo_files_to_weld = ["FrankaPickPlaceSanity_randomdonutonly_zeros", "FrankaPickPlaceSanity_randomdonutonly_zeros"]
bc_files_to_weld = ["real_runs/BC/FrankaPickPlaceSanityCorrections_randomdonutonly_zeros_seed=10_"]
final_directory = "FrankaPickPlaceSanity_randomdonutonly_75eps"
assert len(demo_files_to_weld) > 1, "you can't weld a single file!"

print("loading base")
base_buffer = pkl.load(open(get_out + demo_files_to_weld[0] + pickle_name, "rb")) #the expert replay demo
print(f"The number of current episodes: {len(base_buffer.allEpisodes)}")

for file in demo_files_to_weld[1:]:
    print(f"loading {file}")
    to_add = pkl.load(open(get_out + file + pickle_name, "rb")) #second expert replay demo
    base_buffer.update(to_add)
    print(f"The number of current episodes now: {len(base_buffer.allEpisodes)}")

for file in bc_files_to_weld:
    print(f"loading {file}")
    to_add = pkl.load(open(file + pickle_name, "rb")) #interventions demo
    base_buffer.update(to_add)
    print(f"The number of current episodes now: {len(base_buffer.allEpisodes)}")
    
print("making final directory")
try:
    os.mkdir(get_out + final_directory)
except:
    print("folder already exists!")
    
print("dumping!")    
pkl.dump(base_buffer, open(get_out + final_directory + pickle_name, "wb" ), protocol=4 )
print("done! ")