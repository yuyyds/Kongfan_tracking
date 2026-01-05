"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages    # 递归导入包中的所有子模块

##
# Register Gym environments.
##


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
