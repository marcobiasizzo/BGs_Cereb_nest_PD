#%%
import matplotlib.pyplot as plt

import numpy as np
import pickle as p

# from marco_nest_utils import utils
path = "/home/modelling/Desktop/benedetta/BGs_Cereb_nest_PD/"
name ="shared_results/complete_580ms_x_5_sol17_both_dopa_EBCC_test4_ctx_diff_input_pc_winds_adj"
name1=""
f = open(path + name + "/rasters"+name1,"rb")
rster = p.load(f)
f.close()

f = open(path + name + "/model_dic"+name1,"rb")
model = p.load(f)
f.close()
n_trial = model["trials"]

id = 4

t = rster[id]["times"]
times = t.reshape(len(t),1)
ev = rster[id]["neurons_idx"]
ev = ev.reshape(len(ev),1)
act = np.concatenate([times,ev], axis=1)

#%%

fig = plt.figure(figsize=[12,12])
plt.scatter(rster[id]["times"], rster[id]["neurons_idx"], s =1)
for i in range(n_trial):
    plt.axvline(350+ 580*i, c="red")
    plt.axvline(380+ 580*i, c="red")
plt.xlabel("Time [ms]")
plt.ylabel("Glom idx")
plt.title("Raster Glom")

# %%
fig = plt.figure()
plt.scatter(rster[3]["times"], rster[3]["neurons_idx"], s =10)
# %%
fig = plt.figure(figsize=[12,12])
plt.scatter(rster[2]["times"], rster[2]["neurons_idx"], s =10)
for i in range(n_trial):
    plt.axvline(350+ 580*i, c="red")
    plt.axvline(380+ 580*i, c="red")
plt.xlabel("Time [ms]")
plt.ylabel("Glom idx")
plt.title("Raster Glom")
# %%



import imageio


num_frames = n_trial

 

# Create an empty list to store the images

images = []
plt.rcParams['figure.dpi'] = 300
 

for i in range(num_frames):
    values = (t<580*(i+1))&(t>580*(i))

    act1 = act[values,:]

    fig, ax = plt.subplots()
    

    ax.scatter(act1[:,0], act1[:,1], s =.05)

    plt.axvline(350+ 580*i, c="red", label = "US")
    plt.axvline(380+ 580*i, c="red")
    plt.title("trial "+str(i+1))
    plt.xlabel("Time [ms]")
    plt.ylabel("Granule ID")
    plt.legend(loc ='upper right')
    #ax.scatter(x, np.cos(x + 2 * np.pi * i / num_frames), color='blue', alpha=0.5)
    fig.canvas.draw()
    # images.append(fig)
    images.append(np.array(fig.canvas.renderer.buffer_rgba()))

    plt.show()

 #%%

# Save the images as a gif

imageio.mimsave('granule_scatter_adj5.gif', images, fps=1)
# %%
