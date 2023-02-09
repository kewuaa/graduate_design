from ..config import config_for_data
from . import loader


print('init data...')
loader.init(
    config_for_data.image_num,
    config_for_data.image_size,
    config_for_data.min_circle_num,
    config_for_data.max_circle_num,
    config_for_data.min_circle_size,
    config_for_data.max_circle_size,
    config_for_data.theta_step,
    config_for_data.start_angle,
    config_for_data.end_angle,
)
print('done')
