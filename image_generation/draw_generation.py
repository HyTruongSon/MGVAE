import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from PIL import Image
import PIL

'''
def draw(directory, filename, output):
    gener_set = np.load(directory + "/" + filename)
    num_samples = gener_set.shape[0]
    for sample in range(num_samples):
        image = gener_set[sample, :, :, :]
        plt.clf()
        if image.shape[2] == 1:
            plt.imshow(np.squeeze(image, axis = 2), cmap = 'gray')
        else:
            plt.imshow(image)
        plt.savefig(directory + "/" + output + "/generation_" + str(sample) + ".png")
        if sample % 100 == 0:
            print('Done', sample)
'''

def draw(directory, filename, output, num_rows = 10, num_cols = 10, num_figures_to_draw = 10):
    gener_set = np.load(directory + "/" + filename)
    num_samples = gener_set.shape[0]
    height = gener_set.shape[1]
    width = gener_set.shape[2]
    color_channels = gener_set.shape[3]
    distance = 2
    figure = np.ones((height * num_rows + distance * (num_rows + 1), width * num_cols + distance * (num_cols + 1), color_channels))
    sample = 0
    num_figures = 0
    while sample + num_rows * num_cols <= num_samples:
        # Option 1
        '''
        plt.clf()
        plt.tight_layout()
        gs = gridspec.GridSpec(num_rows, num_cols)
        gs.update(wspace = 0.05, hspace = 0.05) # set the spacing between axes.
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                image = gener_set[sample + idx, :, :, :]
                plt.subplot(num_rows, num_cols, idx + 1)
                plt.axis('off')
                if color_channels == 1:
                    plt.imshow(np.squeeze(image, axis = 2), cmap = 'gray')
                else:
                    plt.imshow(image)
        plt.subplots_adjust(wspace = 0.01, hspace = 0.01)
        plt.savefig(directory + "/" + output + "/generation_" + str(num_figures) + ".png")
        '''

        # Option 2
        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * 10 + j
                image = gener_set[sample + idx, :, :, :] * 255
                x1 = height * i + distance * (i + 1)
                x2 = x1 + height
                y1 = width * j + distance * (j + 1)
                y2 = y1 + width
                figure[x1:x2, y1:y2, :] = image[:, :, :]

        '''
        plt.clf()
        plt.axis('off')
        image_fn = directory + "/" + output + "/generation_" + str(num_figures) + ".png"
        if color_channels == 1:
            plt.imsave(image_fn, figure.squeeze(2), cmap = 'Greys')
        else:
            plt.imsave(image_fn, figure)
        '''
        
        figure = figure.astype('uint8')
        if color_channels == 1:
            PIL_image = Image.fromarray(figure.squeeze(2))
        else:
            PIL_image = Image.fromarray(figure).convert('RGB')
        PIL_image.save(directory + "/" + output + "/generation_" + str(num_figures) + ".png")

        sample += num_rows * num_cols
        num_figures += 1
        if sample % 100 == 0:
            print('Done', sample)
        if num_figures >= num_figures_to_draw:
            break


directory = 'train_mgvae_conv-mnist'
filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_2.npy'
output = 'generation_32x32'

draw(directory, filename, output, 12, 12)

filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_1.npy'
output = 'generation_16x16'

draw(directory, filename, output, 12, 12)

filename = 'train_mgvae_conv.mnist.num_epoch.256.batch_size.20.learning_rate.0.001.kl_loss.1.seed.123456789.cluster_height.2.cluster_width.2.n_levels.2.n_layers.4.Lambda.1.hidden_dim.256.z_dim.256.resolution_level_0.npy'
output = 'generation_8x8'

draw(directory, filename, output, 12, 12)

