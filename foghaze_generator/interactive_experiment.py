from asm_foghaze_generator import ASMFogHazeGenerator
from midas_dmap_estimator import MidasDmapEstimator
from tkinter import Tk, Button, filedialog, messagebox
import matplotlib.pyplot as plt
import numpy as np
import os
import time


class Experiment2:
    def print_img_info(self, print_img: bool = False):
        img = self._rgb_image

        print(
            'Image information:',
            f'(height, width, channel): {img.shape}',
            f'Dtype: {img.dtype}',
            f'Min value: {np.amin(img)}',
            f'Max value: {np.amax(img)}',
            sep='\n')
        
        if print_img:
            print(img)
        print()


    def save_fh_image(self, fh_image: np.ndarray):
        *dir_path, file_name = self.file_path
        fh_name = f'{file_name.split(".")[0]}_fh.jpg'
        fh_filepath = '/'.join(dir_path + [fh_name])

        if os.path.exists(fh_filepath):
            os.remove(fh_filepath)
        plt.imsave(fh_filepath, fh_image)


    def save_idmap(self, dmap: np.ndarray, cmap='gray'):
        *dir_path, file_name = self.file_path
        idmap_name = f'{file_name.split(".")[0]}_{cmap}.jpg'
        idmap_filepath = '/'.join(dir_path + [idmap_name])

        if os.path.exists(idmap_filepath):
            os.remove(idmap_filepath)
        plt.imsave(idmap_filepath, dmap, cmap=cmap)

    
    def _load_image_info(self):
        idmap = self._fh_generator.inverse_dmaps


    def plot_perlin_noise(self):
        fig, axs = plt.subplots(1, 2)



    """
    pair: a tuple of (clear RGB image, foggy-hazy RGB image)
    interactive_params: configurations for interactive params. Format is like:
    {
        'param_name': {
            'current_val': int | float | np.ndarray - if np.ndarray, will display average
            'bounds': tuple (low, high),
            'step': int | float
        }
    }
    fh_generator: the foghaze generator
    """
    def plot_interactive_pair(pair: tuple, interactive_params: dict, fh_generator: ASMFogHazeGenerator):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(pair[0])
        axs[0].set_title('Clear')

        fh_axs = axs[1].imshow(pair[1])
        axs[1].set_title('Foggy-Hazy')

        sliders = {}
        param_counts = len(interactive_params)
        bottom_offset = 0.05 * (param_counts + 1) 
        for key, config in interactive_params.items():
            bottom_offset -= 0.05
            valmin, valmax = config['bounds']

            sliders[key] = Slider(
                fig.add_axes([0.5, bottom_offset, 0.4, 0.05]),
                label = key,
                valmin = valmin, valmax = valmax,
                valinit = config['current_val'], 
                valstep = config['step']
            )
        
        submit_btn = Button(fig.add_axes([0.8, 0.015, 0.1, 0.04]), 'Submit')
        # text_box = plt.text(0, 0, 'Hello', transform=fig.add_axes([0.1, 0.4, 0.4, 0.1]).transAxes)

        def _update_fh_image(event):
            new_val = {}
            for key, slider in sliders.items():
                new_val[key] = slider.val
            
            fh_generator.atm_lights = [new_val['A']]
            fh_generator.scattering_coefs = [new_val['Beta']]
            new_fh = fh_generator.generate_foghaze_images()[0]

            fh_axs.set_data(new_fh)
            fig.canvas.draw_idle()
            print('Done updating image.')

        submit_btn.on_clicked(_update_fh_image)

        if param_counts > 3:
            fig.subplots_adjust(bottom=(0.15 * (param_counts - 3)))

        plt.show()




class InteractiveExperiment:
    _fh_generator: ASMFogHazeGenerator
    _rgb_image: np.ndarray
    file_path: str


    def __init__(self, fh_generator: ASMFogHazeGenerator):
        self._fh_generator = fh_generator


    def input_rgb_image(self):
        file_path = filedialog.askopenfilename(
            title='Select Image',
            filetypes=(('Image File', '*.jpg *.jpeg'), ('All Files', '*.*'))
        )

        if file_path:
            image = plt.imread(file_path)
            self._rgb_image = image

            plt.figure(1, clear=True)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Input RGB Image {image.shape}')
            plt.show()


    def configure_opmode(self):
        plt.figure(2, clear=True)
        plt.plot([1, 40, 15, 35], [1, 45, 27, 64])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mode 2')
        plt.show()


    def configure_params(self):
        plt.figure(3, clear=True)


    def exec_generator(self):
        plt.figure(4, clear=True)

    
    def show_clear(self):
        try:
            clear = self._fh_generator.rgb_images[0]
            plt.figure(5, clear=True)
            plt.imshow(clear)
            plt.show()
        except:
            messagebox.showerror('Error', 'No clear image to display!')

    
    def show_fh(self):
        try:
            fh = self._fh_generator.fh_images[0]
            plt.figure(6, clear=True)
            plt.imshow(fh)
            plt.show()
        except:
            messagebox.showerror('Error', 'No foggy-hazy image to display!')


    def show_idmap(self):
        try:
            idmap = self._fh_generator.inverse_dmaps[0]
            plt.figure(7, clear=True)
            plt.imshow(idmap)
            plt.show()
        except:
            messagebox.showerror('Error', 'No inverse depth map to display!')


    def show_pnoise_beta(self):
        try:
            plt.figure(8, clear=True)
            print('Show Perlin noise')
        except:
            messagebox.showerror('Error', 'No Perlin noise of Beta to display!')


    def run(self):
        root = Tk()
        root.geometry('500x300')
        buttons = []

        buttons.append(Button(root, text='Input RGB Image', command=self.input_rgb_image))
        buttons.append(Button(root, text='Configure Operation Mode', command=self.configure_opmode))
        buttons.append(Button(root, text='Configure Parameters', command=self.configure_params))
        buttons.append(Button(root, text='Execute Generator', command=self.exec_generator))
        buttons.append(Button(root, text='Clear Image', command=self.show_clear))
        buttons.append(Button(root, text='Foggy-Hazy Image', command=self.show_fh))
        buttons.append(Button(root, text='Inverse Depth Map', command=self.show_idmap))
        buttons.append(Button(root, text='Perlin Noise of Beta', command=self.show_pnoise_beta))

        for i, btn in enumerate(buttons):
            if i < 4:
                btn.grid(row=i, column=0, padx=20, pady=20)
            else:
                btn.grid(row=i-4, column=1, padx=20, pady=20)

        root.mainloop()


if __name__ == '__main__':
    fh_generator = ASMFogHazeGenerator(MidasDmapEstimator())
    experiment = InteractiveExperiment(fh_generator)
    experiment.run()
