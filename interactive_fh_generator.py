from foghaze_generator.asm_foghaze_generator import ASMFogHazeGenerator, ATM_LIGHT_OPMODES, SCATTERING_COEF_OPMODES
from matplotlib.widgets import TextBox, Button as PltButton
from foghaze_generator.midas_dmap_estimator import MidasDmapEstimator
from tkinter import Tk, filedialog, messagebox, Button as TkButton
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from ast import literal_eval
import cv2 as cv
import traceback
from foghaze_generator.helper import get_perlin_noise
from datetime import datetime


def print_img_info(img, print_img: bool = False):
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


class InteractiveFHGenerator:
    _fh_generator: ASMFogHazeGenerator
    file_path: str
    generation_result: tuple
    _info_figure_ids = {} # list of figures used to display information


    def __init__(self, fh_generator: ASMFogHazeGenerator):
        self._fh_generator = fh_generator


    def _get_time_suffix_filepath(self, file_path):
        part = os.path.splitext(file_path)
        file_path = part[0] + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + part[1]

        return file_path

    
    def save_fh_image(self):
        try:
            fh_image = self._fh_generator.fh_images[0]
            *dir_path, file_name = self.file_path.split('/')
            fh_name = f'{file_name.split(".")[0]}_fh.jpg'
            fh_filepath = '/'.join(dir_path + [fh_name])
            fh_filepath = self._get_time_suffix_filepath(fh_filepath)

            plt.imsave(fh_filepath, fh_image)
            messagebox.showinfo('Save Foggy-Hazy Image', f'Save image successfully at {fh_filepath}!')
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('Error', str(e))


    def save_idmap(self, cmap='gray'):
        try:
            idmap = self._fh_generator.inverse_dmaps[0]
            *dir_path, file_name = self.file_path.split('/')
            idmap_name = f'{file_name.split(".")[0]}_{cmap}.jpg'
            idmap_filepath = '/'.join(dir_path + [idmap_name])
            idmap_filepath = self._get_time_suffix_filepath(idmap_filepath)

            plt.imsave(idmap_filepath, idmap, cmap=cmap)
            messagebox.showinfo('Save Inverse Depth Map', f'Save inverse depth map successfully at {idmap_filepath}!')
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('Error', str(e))

    
    def perlin_noise_image(self):
        clear = self._fh_generator.rgb_images[0]
        pnoise_config = self._fh_generator.pnoise_configs[0]
        pnoise = get_perlin_noise(clear.shape, pnoise_config, (0, 255)).astype(np.uint8)
        pnoise_grayscale = pnoise[:, :, 0]

        return pnoise_grayscale


    def input_rgb_image(self):
        file_path = filedialog.askopenfilename(
            title='Select RGB Image',
            filetypes=(('Image File', '*.jpg *.jpeg'), ('All Files', '*.*'))
        )

        if file_path:
            try:
                self.file_path = file_path
                image = plt.imread(file_path)
                self._fh_generator.rgb_images = [image]
                plt.figure(1, clear=True)
                plt.title(f'Input Image {image.shape}')
                plt.axis('off')
                plt.imshow(image)
                plt.show()
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror('Error', str(e))


    def input_inverse_dmap(self):
        file_path = filedialog.askopenfilename(
            title='Select Depth Map',
            filetypes=(('Image File', '*.jpg *.jpeg'), ('All Files', '*.*'))
        )

        if file_path:
            try:
                idmap = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                self._fh_generator.inverse_dmaps = [idmap]
                plt.figure(2, clear=True)
                plt.title(f'Input Image {idmap.shape}')
                plt.axis('off')
                plt.imshow(idmap, cmap='gray')
                plt.show()
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror('Error', str(e))


    def configure_opmode(self):
        current_opmode = self._fh_generator.operation_mode
        plt.figure(3, clear=True)

        axis_A = plt.axes([0.4, 0.6, 0.3, 0.05])
        text_box_A = TextBox(axis_A, 'Atmospheric Light', initial=f'{current_opmode["atm_light"]}')
        axis_A.text(0, 1.3, f'Valid values: {ATM_LIGHT_OPMODES}', color='red')

        axis_beta = plt.axes([0.4, 0.4, 0.3, 0.05])
        text_box_beta = TextBox(axis_beta, 'Scattering Coefficient', initial=f'{current_opmode["scattering_coef"]}')
        axis_beta.text(0, 1.3, f'Valid values: {SCATTERING_COEF_OPMODES}', color='red')

        submit_button = PltButton(plt.axes([0.4, 0.2, 0.2, 0.1]), 'Submit')

        def submit_callback(event):
            new_A_mode = text_box_A.text
            new_beta_mode = text_box_beta.text

            if new_A_mode not in ATM_LIGHT_OPMODES or new_beta_mode not in SCATTERING_COEF_OPMODES:
                raise ValueError('Wrong configuration value!')
            
            self._fh_generator.operation_mode['atm_light'] = new_A_mode
            self._fh_generator.operation_mode['scattering_coef'] = new_beta_mode
            print('Successfully configure operation mode!')

        submit_button.on_clicked(submit_callback)
        plt.show()


    def configure_params(self):
        plt.figure(4, clear=True)

        atm_light = self._fh_generator.atm_lights[0] if self._fh_generator.atm_lights else None
        beta = self._fh_generator.scattering_coefs[0] if self._fh_generator.scattering_coefs else None
        pnoise_config = self._fh_generator.pnoise_configs[0] if self._fh_generator.pnoise_configs else {}
        pnoise_config = {
            'octaves': pnoise_config.get('octaves', 1),
            'persistence': pnoise_config.get('persistence', 0.5),
            'lacunarity': pnoise_config.get('lacunarity', 2.0),
            'repeatx': pnoise_config.get('repeatx', 1024),
            'repeaty': pnoise_config.get('repeaty', 1024),
            'base': pnoise_config.get('base', 0),
            'scale': pnoise_config.get('scale', 1)
        }

        bottom_offset = 0.3
        left_offset = 0.25
        input_width = 0.5
        input_height = 0.05

        tb_pnoise = {}
        for key, value in reversed(pnoise_config.items()):
            tb_pnoise[key] = TextBox(plt.axes([left_offset, bottom_offset, input_width, input_height]), key.capitalize(), initial=value)
            bottom_offset += 0.06
        
        axis_beta = plt.axes([left_offset, bottom_offset + 0.03, input_width, input_height])
        tb_beta = TextBox(axis_beta, 'Scattering Coefficient', initial=beta)
        axis_beta.text(0, 1.2, f'Valid type: float | tuple[float, float]', color='red')

        axis_A = plt.axes([left_offset, bottom_offset + 0.15, input_width, input_height])
        tb_A = TextBox(axis_A, 'Atmospheric Light', initial=atm_light)
        axis_A.text(0, 1.2, f'Valid type: int | tuple[int, int]', color='red')

        submit_button = PltButton(plt.axes([0.4, 0.1, 0.2, 0.1]), 'Submit')

        def submit_callback(event):
            new_A = literal_eval(tb_A.text) if tb_A.text else None
            new_A = None if new_A == -1 else new_A
            self._fh_generator.atm_lights = [new_A]

            new_beta = literal_eval(tb_beta.text) if tb_beta.text else None
            new_A = None if new_beta == -1.0 else new_beta
            self._fh_generator.scattering_coefs = [new_beta]

            new_pnoise_config = {}
            for key, textbox in reversed(tb_pnoise.items()):
                new_pnoise_config[key] = literal_eval(textbox.text)
            self._fh_generator.pnoise_configs = [new_pnoise_config]

            print('Configure parameters successfully!')

        submit_button.on_clicked(submit_callback)

        plt.show()


    def exec_generator(self):
        try:
            for fig_id in plt.get_fignums():
                if fig_id in self._info_figure_ids:
                    self._info_figure_ids[fig_id]() # call to update info figure (figure displaying information)

            self.generation_result = self._fh_generator.generate_foghaze_images()[0]
            messagebox.showinfo('Execute Generator', 'Done')
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror('Error', str(e))

    
    def show_clear_and_fh(self):
        try:
            clear = self._fh_generator.rgb_images[0]
            fh, _, real_atm_light, real_beta = self.generation_result
            fig_id = 5
            fig, axs = plt.subplots(1, 2, num=fig_id, clear=True)
            self._info_figure_ids[fig_id] = self.show_clear_and_fh

            if type(real_atm_light) is np.ndarray:
                real_atm_light = np.mean(real_atm_light)
            if type(real_beta) is np.ndarray:
                real_beta = np.mean(real_beta)

            axs[0].set_title(f'Input Image {clear.shape}')
            axs[0].imshow(clear)
            axs[0].axis('off')

            axs[1].set_title(f'Foggy-Hazy Image (A = {real_atm_light}, β = {real_beta})')
            axs[1].imshow(fh)
            axs[1].axis('off')

            plt.show()
        except:
            traceback.print_exc()
            messagebox.showerror('Error', 'No available generation result! Please execute generator first!')

        
    def show_idmap_and_pnoise(self):
        try:
            idmap = self.generation_result[1]
            real_beta = self.generation_result[3]
            fig_id = 6
            fig, axs = plt.subplots(1, 2, num=fig_id, clear=True)
            self._info_figure_ids[fig_id] = self.show_idmap_and_pnoise

            axs[0].set_title(f'Inverse Depth Map {idmap.shape}')
            axs[0].imshow(idmap, cmap='gray')
            axs[0].axis('off')

            axs[1].set_title(f'Perlin Noise of β (avg. β = {np.mean(real_beta)})')
            axs[1].imshow(self.perlin_noise_image(), cmap='gray')
            axs[1].axis('off')

            plt.show()
        except:
            traceback.print_exc()
            messagebox.showerror('Error', 'No available generation result! Please execute generator first!')

    
    def show_overall_info(self):
        try:
            clear = self._fh_generator.rgb_images[0]
            opmode = self._fh_generator.operation_mode
            fh, idmap, real_atm_light, real_beta = self.generation_result

            if type(real_atm_light) is np.ndarray:
                real_atm_light = np.mean(real_atm_light)
            if type(real_beta) is np.ndarray:
                real_beta = np.mean(real_beta)

            fig_id = 7
            fig, axs = plt.subplots(2, 2, num=fig_id, clear=True)
            self._info_figure_ids[fig_id] = self.show_overall_info

            axs[0][0].set_title(f'Input Image {clear.shape}')
            axs[0][0].imshow(clear)
            axs[0][0].axis('off')

            axs[0][1].set_title(f'Foggy-Hazy Image (A = {real_atm_light}, β = {real_beta})')
            axs[0][1].imshow(fh)
            axs[0][1].axis('off')

            axs[1][0].set_title(f'Inverse Depth Map {idmap.shape}')
            axs[1][0].imshow(idmap, cmap='gray')
            axs[1][0].axis('off')

            if opmode['scattering_coef'] == 'pnoise':
                axs[1][1].set_title('Perlin Noise of β')
                axs[1][1].imshow(self.perlin_noise_image(), cmap='gray')
                axs[1][1].axis('off')

            plt.show()
        except:
            traceback.print_exc()
            messagebox.showerror('Error', 'No available generation result! Please execute generator first!')
    

    def run(self):
        root = Tk()
        buttons = []

        buttons.append(TkButton(root, text='Input RGB Image (*)', command=self.input_rgb_image))
        buttons.append(TkButton(root, text='Input Gray Inverse Depth Map', command=self.input_inverse_dmap))
        buttons.append(TkButton(root, text='Configure Operation Mode', command=self.configure_opmode))
        buttons.append(TkButton(root, text='Configure Parameters', command=self.configure_params))
        buttons.append(TkButton(root, text='Execute Generator', command=self.exec_generator))
        buttons.append(TkButton(root, text='Save Foggy-Hazy Image', command=self.save_fh_image))
        buttons.append(TkButton(root, text='Save Inverse Depth Map', command=self.save_idmap))
        buttons.append(TkButton(root, text='Show Clear & Foggy-Hazy Image', command=self.show_clear_and_fh))
        buttons.append(TkButton(root, text='Show Depth Map & Perlin Noise', command=self.show_idmap_and_pnoise))
        buttons.append(TkButton(root, text='Show Generation Result', command=self.show_overall_info))

        for i, btn in enumerate(buttons):
            if i < 5:
                btn.grid(row=i, column=0, padx=20, pady=20)
            else:
                btn.grid(row=i-5, column=1, padx=20, pady=20)

        def on_close():
            plt.close('all')
            root.destroy()

        root.protocol('WM_DELETE_WINDOW', on_close)
        root.mainloop()


if __name__ == '__main__':
    fh_generator = ASMFogHazeGenerator(MidasDmapEstimator())
    interactive_program = InteractiveFHGenerator(fh_generator)
    interactive_program.run()
