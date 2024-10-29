#%%
import numpy as np
from pathlib import Path
import cv2

import os
import sys
module_path = os.path.abspath(os.path.join('/home/vbauer/MEGA/Master/OIE/Thesis 2024/src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
module_path = os.path.abspath(os.path.join('/home/vbauer/MEGA/Master/OIE/Thesis 2024/src/camera_modelling/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from camera_modelling import raw_processing, geometry, utils, image_operations
from camera_modelling.camera import Camera, SensorParameters, IntrinsicCameraParameters, ExtrinsicCameraParameters

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

#%%

sensor_params = SensorParameters.from_json_file('/home/vbauer/MEGA/Master/OIE/Thesis 2024/data/imx708_sensor_parameters.json')

output_dir = Path(f'../../data/sky-images/')
if not output_dir.exists():
    output_dir.mkdir(parents=True)
    
image_output_size = (880, 880)

#%%

for camera_name in ['ikarus', 'daedalus']:
    
    if not (output_dir / camera_name).exists():
        (output_dir / camera_name).mkdir(parents=True)
    
    if camera_name == 'ikarus':
        intrinsic_params = IntrinsicCameraParameters.from_pyOCamCalib_file(
            '/home/vbauer/MEGA/Master/OIE/Thesis 2024/data/calibration/intrinsic/ikarus/calibration-ikarus_2024-05-23_11-17-56/calibration_0.76_lens_calibration-ikarus_2024-05-23_11-17-56.json'
        )
    elif camera_name == 'daedalus':
        intrinsic_params = IntrinsicCameraParameters.from_pyOCamCalib_file(
            '/home/vbauer/MEGA/Master/OIE/Thesis 2024/data/calibration/intrinsic/daedalus/calibration-daedalus-2024-05-29_11-27-10/calibration_0.76_lens_calibration-daedalus-2024-05-29_11-27-10.json'
        )
    else:
        raise ValueError(f'Unknown camera name: {camera_name}')

    extrinsic_calibration_dir = Path('/home/vbauer/MEGA/Master/OIE/Thesis 2024/data/calibration/extrinsic/')

    extrinsic_params = ExtrinsicCameraParameters.from_json_file(
        extrinsic_calibration_dir / camera_name / 'extrinsic_calibration_parameters.json'
    )

    camera = Camera(sensor_params, intrinsic_params, extrinsic_params)

    exposure_stack_dir = Path(f'/home/vbauer/MEGA/Master/OIE/Thesis 2024/data/sky-images/exposure_stacks/{camera_name}')

    all_img_ids = { f.stem for f in exposure_stack_dir.glob('*.npz') }
    
    # keep only images that are between 6:00 and 16:00
    # super hacky way to do this, but it works...
    daytime_img_ids = set(sorted([ img_id for img_id in all_img_ids if '05-59' <= img_id.split('T')[1][0:5] and img_id.split('T')[1][0:5] <= '16-01' ]))


    # remove any img_ids that are already processed
    pending_img_ids =  { img_id for img_id in daytime_img_ids if not (output_dir / {camera_name} / f'{camera_name}_{img_id}.jpg').exists() }
    
    image_ids = {
        'all': all_img_ids,
        'daytime': daytime_img_ids,
        'pending': pending_img_ids,
        'processed': set(daytime_img_ids) - set(pending_img_ids)
    }
    
    # show a quick summary of the images that will be processed
    print(f'Camera: {camera_name}')
    print(f'All images: {len(all_img_ids)}')
    print(f'Daytime images: {len(daytime_img_ids)}')
    print(f'Already processed images: {len(daytime_img_ids) - len(pending_img_ids)}')
    print(f'Pending images: {len(pending_img_ids)}')

    # reproject the HDR images to an equidistant projection in camera space
    # we reproject to a common projection, so that the images can be compared more easily
    # equidistant, because it is a simple projection and make the distortion more uniform than the unprojected image
    # camera space, because we don't care about the camera's orientation in the world, rather we care about the view of the sky from the camera's perspective to classify the artifacts
    # also, the camera space is not affected by errors in the extrinsic calibration, and we know there are some errors in the extrinsic calibration 
    reprojection_map = camera.generate_sky_projection_map(width=image_output_size[0], height=image_output_size[1], projection='equidisant', target_space='camera')
    
    def process_image(img_id):
        input_file_path = exposure_stack_dir / f'{img_id}.npz'
        output_file_path = output_dir / {camera_name} / f'{camera_name}_{img_id}.jpg'

        exposure_stack_raw_data, stack_metadata = utils.load_exposure_stack(input_file_path)
        exposure_times = np.array([ meta_data['ExposureTime'] for meta_data in stack_metadata['metadata_list'] ])

        colour_gains_rb = np.array([ meta_data['ColourGains'] for meta_data in stack_metadata['metadata_list'] ]) # red and blue gains, green gain is 1 implicitly
        colour_gains_rgb = np.insert(colour_gains_rb, 1, values=1, axis=1) # red, green, blue gains, added green gain as 1 explicitly
        # print(f'Colour gains: {colour_gains_rgb}')
        color_correction_matrix = np.array(stack_metadata['metadata_list'][0]['ColourCorrectionMatrix']).reshape(3, 3)

        # create a color hdr image and a mask of over- and underexposed pixels
        hdr_image_color, o, u = raw_processing.compose_hdr_color_image(exposure_stack_raw_data, exposure_times, sensor_params.bayer_pattern, colour_gains_rgb, color_correction_matrix, sensor_params.black_level, sensor_params.white_level)
        # mask out all pixels that are below 90° incident angle for the camera
        hdr_image_color = camera.crop_to_sky_disk(hdr_image_color, mask_value=0.0)
        hdr_image_color_unit8 = (hdr_image_color * 255).astype(np.uint8)
        # reproject the image to the equidistant projection in camera space
        reprojected_image = cv2.remap(hdr_image_color_unit8, reprojection_map, None, cv2.INTER_CUBIC)

        # jpg compression with 95% quality
        cv2.imwrite(str(output_file_path), cv2.cvtColor(reprojected_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 99])
        
        image_ids['processed'].add(img_id)
        image_ids['pending'].remove(img_id)
        
    with tqdm_joblib(tqdm(desc=f'Processing images for camera: {camera_name}', total=len(pending_img_ids))) as progress_bar:
        joblib.Parallel(n_jobs=10)(joblib.delayed(process_image)(img_id) for img_id in pending_img_ids)
        
    print(f'Processed images: {len(pending_img_ids)}')
    print(f'Pending images: {len(pending_img_ids)}')

print('Done')

# %%
