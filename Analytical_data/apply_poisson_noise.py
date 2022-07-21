import click
import itk
import numpy as np

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--input', '-i', required = True)
@click.option('--output', '-o', required = True)
def apply_poisson_noise_click(input, output):
    apply_poisson_noise(input, output)




def apply_poisson_noise(input, output):
    print(f'Opening {input}...')

    projection_itk = itk.imread(input)
    vSpacing = np.array(projection_itk.GetSpacing())
    vOffset = np.array(projection_itk.GetOrigin())

    projection_array = itk.array_from_image(projection_itk)


    noisy_projection_array = np.random.poisson(lam = projection_array, size = projection_array.shape).astype(float)

    noisy_projection_image = itk.image_from_array(noisy_projection_array)
    noisy_projection_image.SetSpacing(vSpacing)
    noisy_projection_image.SetOrigin(vOffset)
    itk.imwrite(noisy_projection_image,output)

    print(f'Done! Output at {output}')


if __name__=='__main__':
    apply_poisson_noise_click()