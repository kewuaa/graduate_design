from ..config import config_for_data
from .dataset import init


print('init data...')
init(
    config_for_data.image_num,
    config_for_data.image_size,
    config_for_data.pixel,
    config_for_data.circle_num,
    config_for_data.circle_size,
    config_for_data.theta_step,
    config_for_data.angle,
)
print('done')
